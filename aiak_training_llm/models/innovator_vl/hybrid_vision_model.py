"""VisionTransformer module"""

import functools
from typing import Optional, Tuple

import torch
import torch.nn as nn
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
)

from aiak_training_llm.models.innovator_vl.dinov3_layer_specs import (
    get_dinov3_layer_with_transformer_engine_spec
)
from aiak_training_llm.models.innovator_vl.dinov3_vit_model import DINOv3ViTModel
from aiak_training_llm.models.innovator_vl.innovator_vl_config import VisionConfig
from aiak_training_llm.models.innovator_vl.vision_model import RiceViTModel


def get_sliglip2_model():
    config = TransformerConfig(
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        ffn_hidden_size=4304,
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        activation_func=functools.partial(nn.functional.gelu, approximate="tanh"),
    )

    spec = get_vit_layer_with_transformer_engine_spec()
    return CLIPViTModel(
        config,
        spec,
        add_class_token=False,
        class_token_len=0,
        img_h=384,
        img_w=384,
        model_subtype="siglip",
    )


def disable_k_bias_for_layer(layer):
    attn = layer.self_attention
    qkv = getattr(attn, "query_key_value", None)
    if qkv is None or qkv.bias is None:
        return

    # 计算各段长度（兼容 MHA 与 MQA/GQA）
    num_heads = attn.num_attention_heads
    # hidden_size = attn.hidden_size  # 某些版本可直接拿
    # 更稳妥地用 head_dim 推导
    # 对多数实现：head_dim = hidden_size // num_heads
    # 某些实现直接暴露 kv_channels
    head_dim = getattr(attn, "kv_channels", None)
    if head_dim is None:
        hidden_size = layer.config.hidden_size
        head_dim = hidden_size // num_heads

    num_kv_heads = getattr(attn, "num_kv_heads", num_heads)  # MHA 时等于 num_heads

    q_len = num_heads * head_dim
    k_len = num_kv_heads * head_dim
    k_start, k_end = q_len, q_len + k_len

    with torch.no_grad():
        qkv.bias[k_start:k_end].zero_()
    qkv.bias[k_start:k_end].requires_grad_(False)


def get_dinov3_model():
    config = TransformerConfig(
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        layernorm_epsilon=1e-5,
        attention_dropout=0.0,
    )
    spec = get_dinov3_layer_with_transformer_engine_spec()

    model = DINOv3ViTModel(
        config,
        spec,
        patch_dim=16,
        img_h=512,
        img_w=512,
        layerscale_value=1.0,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
    )
    # for dinov3 config, "key_bias": false
    for lyr in model.transformer.layers:
        disable_k_bias_for_layer(lyr)

    return model



class HybridVisionModel(VisionModule):
    """
    Hybrid vision model that runs RiceViT, SigLIP2, and DINOv3 encoders.
    Returns features from all three encoders for fusion in the adapter.
    """

    def __init__(
        self, config: VisionConfig, transformer_layer_spec: ModuleSpec
    ) -> None:
        super().__init__(config=config)
        self.freeze_external = config.freeze_external

        # Initialize three encoders
        self.rice_vit = RiceViTModel(config, transformer_layer_spec)
        self.siglip_model = get_sliglip2_model()
        self.dinov3_model = get_dinov3_model()

        if config.freeze_external:
            for module in self.siglip_model:
                for param in module.parameters():
                    param.requires_grad = False

            for module in self.dinov3_model:
                for param in module.parameters():
                    param.requires_grad = False

    def _extract_sequence_features(
        self,
        model: nn.Module,
        pixel_values: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        """Extract sequence features from an encoder model."""
        model = model.to(target_device)
        features = model(pixel_values)

        # For DINOv3: skip CLS token (1) and register tokens (4) to get pure patch features.
        # Reference: preprocessor.py dinov3_fwd()
        if hasattr(model, "config") and hasattr(model.config, "num_register_tokens"):
            num_skip = 1 + model.config.num_register_tokens  # CLS + registers
            features = features[:, num_skip:, :]

        if features.ndim == 2:
            features = features.unsqueeze(1)

        return features.to(target_dtype)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        pixel_values_images_siglip: Optional[torch.Tensor] = None,
        pixel_values_images_dinov3: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Forward pass that returns features from all three encoders.

        Args:
            x: Input tensor for RiceViT [batch, channels, height, width]
            grid_thw: Grid dimensions [num_images, 3]
            pixel_values_images_siglip: Optional SigLIP2 input images
            pixel_values_images_dinov3: Optional DINOv3 input images

        Returns:
            Tuple of:
                - ricevit_features: [total_tokens, hidden_size]
                - window_index: Window index for reordering
                - siglip_features: [batch, num_tokens, hidden_size] or None
                - dinov3_features: [batch, num_tokens, hidden_size] or None
        """
        # Get RiceViT features (base encoder)
        ricevit_features, window_index = self.rice_vit(x, grid_thw, **kwargs)

        # Get SigLIP2 features if provided
        siglip_features = None
        if pixel_values_images_siglip is not None:
            siglip_features = self._extract_sequence_features(
                self.siglip_model,
                pixel_values_images_siglip,
                target_dtype=ricevit_features.dtype,
                target_device=ricevit_features.device,
            )

        # Get DINOv3 features if provided
        dinov3_features = None
        if pixel_values_images_dinov3 is not None:
            dinov3_features = self._extract_sequence_features(
                self.dinov3_model,
                pixel_values_images_dinov3,
                target_dtype=ricevit_features.dtype,
                target_device=ricevit_features.device,
            )

        return ricevit_features, window_index, siglip_features, dinov3_features
