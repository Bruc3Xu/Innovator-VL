# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""DINOv3 ViT model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


try:
    from megatron.core.extensions.transformer_engine import TENorm

    NORM_IMPL = TENorm
except Exception:
    NORM_IMPL = torch.nn.LayerNorm


def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 2D coordinates of the centers of image patches, normalized to the range [-1, +1].
    The center of each patch is exactly halfway between its top-left and bottom-right corners.

    Args:
        num_patches_h (int): Number of patches along the vertical (height) axis.
        num_patches_w (int): Number of patches along the horizontal (width) axis.
        dtype (torch.dtype): The desired data type of the returned tensor.
        device (torch.device): The device for the returned tensor.

    Returns:
        torch.Tensor: A tensor of shape (height * width, 2), where each row contains the (y, x)
            coordinates of a patch center, normalized to [-1, +1].
    """
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    # (height, width, 2) -> (height * width, 2)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    return coords


def augment_patches_center_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:
    """Apply augmentations to patch center coordinates during training.

    Args:
        coords (torch.Tensor): Patch coordinates of shape (num_patches, 2).
        shift (float, optional): Maximum shift value for uniform shift augmentation.
        jitter (float, optional): Maximum jitter value for log-uniform jitter augmentation.
        rescale (float, optional): Maximum rescale value for log-uniform rescale augmentation.

    Returns:
        torch.Tensor: Augmented coordinates.
    """
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        shift_hw = shift_hw.uniform_(-shift, shift)
        coords = coords + shift_hw

    # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        import numpy as np

        jitter_range = np.log(jitter)
        jitter_hw = torch.empty((1, 2), device=coords.device, dtype=coords.dtype)
        jitter_hw = jitter_hw.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_hw

    # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        import numpy as np

        rescale_range = np.log(rescale)
        rescale_hw = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_hw = rescale_hw.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_hw

    return coords


class DINOv3RoPE2D(nn.Module):
    """2D Rotary Position Embedding for DINOv3.

    This module computes 2D RoPE embeddings based on patch center coordinates.
    Unlike standard 1D RoPE for language models, this uses 2D spatial coordinates.

    Args:
        hidden_size (int): Hidden size of the model.
        num_attention_heads (int): Number of attention heads.
        patch_size (int): Size of each patch.
        rope_theta (float): Base period for RoPE. Default: 10000.0
        pos_embed_shift (float): Shift augmentation value. Default: None
        pos_embed_jitter (float): Jitter augmentation value. Default: None
        pos_embed_rescale (float): Rescale augmentation value. Default: None
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        patch_size: int,
        rope_theta: float = 10000.0,
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale

        # Compute inv_freq for RoPE (head_dim / 4 dimensions for 2D)
        inv_freq = 1.0 / (
            self.rope_theta
            ** torch.arange(0, 1, 4.0 / self.head_dim, dtype=torch.float32)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute 2D RoPE embeddings.

        Args:
            pixel_values (torch.Tensor): Input images of shape (batch, channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings for RoPE.
        """
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device

        # Compute patch center coordinates
        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w, dtype=torch.float32, device=device
        )

        # Apply augmentations during training
        if self.training:
            patch_coords = augment_patches_center_coordinates(
                patch_coords,
                shift=self.pos_embed_shift,
                jitter=self.pos_embed_jitter,
                rescale=self.pos_embed_rescale,
            )

        # Compute angles: (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2)
        angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2)
        # Tile to get full head_dim: (height * width, head_dim)
        angles = angles.tile(2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos.to(dtype=pixel_values.dtype), sin.to(dtype=pixel_values.dtype)


class DINOv3ViTModel(VisionModule):
    """DINOv3 ViT vision model.

    This model implements the DINOv3 vision transformer with the following features:
    - 2D RoPE (Rotary Position Embedding) for spatial encoding
    - Register tokens for improved representation learning
    - LayerScale for stable training of deep models
    - DropPath (Stochastic Depth) for regularization
    - Optional Gated MLP

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        ln_post_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_post.
        patch_dim (int): Image patch size. Default: 14
        img_h (int): Input image height. Default: 336
        img_w (int): Input image width. Default: 336
        num_register_tokens (int): Number of register tokens. Default: 4
        use_gated_mlp (bool): Whether to use gated MLP. Default: False
        layerscale_value (float): Initial LayerScale value. Default: 1e-5
        drop_path_rate (float): Drop path rate. Default: 0.0
        rope_theta (float): Base period for RoPE. Default: 10000.0
        pos_embed_shift (float): Shift augmentation for position embeddings. Default: None
        pos_embed_jitter (float): Jitter augmentation for position embeddings. Default: None
        pos_embed_rescale (float): Rescale augmentation for position embeddings. Default: None
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        ln_pre_impl: Union[ModuleSpec, type] = NORM_IMPL,
        ln_post_impl: Union[ModuleSpec, type] = NORM_IMPL,
        patch_dim: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        num_register_tokens: int = 4,
        use_gated_mlp: bool = False,
        layerscale_value: float = 1e-5,
        drop_path_rate: float = 0.0,
        rope_theta: float = 10000.0,
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = None,
    ) -> None:
        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        self.num_register_tokens = num_register_tokens
        self.use_gated_mlp = use_gated_mlp
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        # CLS token (1) + register tokens + patch tokens
        self.seq_length = 1 + self.num_register_tokens + self.num_patches

        # Patch embedding using Conv2d
        self.patch_embeddings = nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.visual_hidden_size)
        )

        # Register tokens
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, self.num_register_tokens, self.visual_hidden_size)
            )
        else:
            self.register_tokens = None

        # 2D RoPE position embeddings
        self.rope_embeddings = DINOv3RoPE2D(
            hidden_size=self.visual_hidden_size,
            num_attention_heads=transformer_config.num_attention_heads,
            patch_size=self.patch_dim,
            rope_theta=rope_theta,
            pos_embed_shift=pos_embed_shift,
            pos_embed_jitter=pos_embed_jitter,
            pos_embed_rescale=pos_embed_rescale,
        )

        # Layer normalization
        self.ln_pre = None
        self.ln_post = None
        if ln_pre_impl is not None:
            self.ln_pre = build_module(
                ln_pre_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.layernorm_epsilon,
            )
        if ln_post_impl is not None:
            self.ln_post = build_module(
                ln_post_impl,
                config=transformer_config,
                hidden_size=self.visual_hidden_size,
                eps=transformer_config.layernorm_epsilon,
            )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the DINOv3 ViT Model.

        Args:
            x (torch.Tensor): Input data of shape [batch, channels, img_h, img_w].
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): Output after final transformer block of shape [b, s, h].
        """
        batch_size = x.shape[0]

        # Patch embedding: [b, c, h, w] -> [b, hidden, h//patch, w//patch]
        x = self.patch_embeddings(x)
        # [b, hidden, grid_h, grid_w] -> [b, hidden, num_patches]
        x = x.flatten(2)
        # [b, hidden, num_patches] -> [b, num_patches, hidden]
        x = x.transpose(1, 2)

        # Add CLS token and register tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, register_tokens, x], dim=1)
        else:
            x = torch.cat([cls_tokens, x], dim=1)

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"

        # Compute 2D RoPE embeddings (only for patch tokens)
        # cos, sin have shape (num_patches, head_dim)
        cos, sin = self.rope_embeddings(x)

        # Pre-normalization
        if self.ln_pre:
            x = self.ln_pre(x)

        # Transpose for transformer: [b, s, h] -> [s, b, h]
        x = x.permute(1, 0, 2).contiguous()

        # Prepare RoPE embeddings in the format expected by Megatron
        # Standard RoPE format: (q_emb, k_emb) where each is (seq_len, 1, 1, head_dim)
        # For DINOv3, we only have embeddings for patch tokens (not prefix tokens)
        # We need to pad the embeddings to match the full sequence length
        num_prefix_tokens = 1 + self.num_register_tokens

        # Pad cos and sin with zeros for prefix tokens
        # Shape: (num_patches, head_dim) -> (seq_len, head_dim)
        cos_padded = torch.cat([
            torch.zeros(num_prefix_tokens, cos.shape[1], dtype=cos.dtype, device=cos.device),
            cos
        ], dim=0)
        sin_padded = torch.cat([
            torch.zeros(num_prefix_tokens, sin.shape[1], dtype=sin.dtype, device=sin.device),
            sin
        ], dim=0)

        # Reshape to Megatron's expected format: (seq_len, 1, 1, head_dim)
        cos_padded = cos_padded.unsqueeze(1).unsqueeze(1)
        sin_padded = sin_padded.unsqueeze(1).unsqueeze(1)

        # Combine into rotary_pos_emb tuple (q_emb, k_emb)
        # For DINOv3, q and k use the same embeddings
        rotary_pos_emb = (cos_padded, sin_padded)

        # Transformer forward with RoPE
        x = self.decoder(
            x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        # Transpose back: [s, b, h] -> [b, s, h]
        x = x.permute(1, 0, 2).contiguous()

        # Post-normalization
        if self.ln_post:
            x = self.ln_post(x)

        return x

    def get_num_image_embeddings(
        self, disable_vision_class_token: bool = False
    ) -> int:
        """Get the number of image embeddings per image.

        Args:
            disable_vision_class_token (bool): Whether to exclude class token.

        Returns:
            int: Number of image embeddings.
        """
        if disable_vision_class_token:
            return self.seq_length - 1  # Exclude CLS token
        return self.seq_length
