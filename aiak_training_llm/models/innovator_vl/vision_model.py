""" VisionTransformer module """

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType, AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from aiak_training_llm.models.qwen_vl.vision_transformer_block import TransformerBlock
from aiak_training_llm.models.qwen_vl.vision_model import _rotate_half, apply_rotary_pos_emb_vision
from aiak_training_llm.models.innovator_vl.innovator_vl_config import VisionConfig
from megatron.training import print_rank_0

def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(t, freqs, config, cu_seqlens=None, rotary_interleaved=False):
    """Apply rotation to positional embedding."""
    orig_dtype = t.dtype
    t = t.float()
    if cu_seqlens is not None:
        freqs = freqs.squeeze(1)
        cos_ = freqs.cos().float().repeat(1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 2)
    else:
        cos_ = freqs.cos().float().repeat(1, 1, 1, 2)
        sin_ = freqs.sin().float().repeat(1, 1, 1, 2)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return t.to(orig_dtype)


class PatchEmbed(torch.nn.Module):
    """Patch Embedding module."""
    def __init__(
        self,
        patch_size: int = 14,
        # temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        # self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # kernel_size = [temporal_patch_size, patch_size, patch_size]
        # self.proj = torch.nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
        self.proj = torch.nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            # -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(torch.nn.Module):
    """Rotary Position Embedding module."""
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.inv_freq = inv_freq.to(torch.cuda.current_device())

    def forward(self, seqlen: int) -> torch.Tensor:
        """ Forward Pass """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionModel(VisionModule):
    """VisionTransformer model. """
    def __init__(self, 
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.spatial_merge_size = spatial_merge_size

        self.rotary_pos_emb = VisionRotaryEmbedding(config.kv_channels // 2)

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        self.decoder = TransformerBlock(
            config=config,
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

    def rot_pos_emb(self, grid_thw):
        """ rotation position embedding """
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """ forward function """
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(1).unsqueeze(2).float()

        x = self.patch_embed(x)
        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]
        x = self.decoder(x, rotary_pos_emb=rotary_pos_emb, attention_mask=None, attn_mask_type=AttnMaskType.no_mask)
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]
        return x, None


class RiceViTModel(VisionModel):
    """"""
    def __init__(self,
        config: VisionConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
        # window_size: int = 112,
    ) -> None:
        super().__init__(config, transformer_layer_spec, spatial_merge_size)
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = list(range(config.num_layers))
        # self.window_size = window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.register_buffer('class_embedding', torch.randn(config.hidden_size))
        self.register_buffer('class_pos_emb', torch.randn(1, config.kv_channels // 2))
        # self.class_embedding = torch.nn.Parameter(torch.randn(config.hidden_size))
        # self.class_pos_emb = torch.nn.Parameter(torch.randn(1, config.kv_channels // 2))

        self.pre_layernorm = torch.nn.LayerNorm(
            config.hidden_size,
            eps=1e-4)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):
            new_x.append(class_tokens[i:i+1])
            new_x.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)

        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i:i+1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)

        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # Start from 0
        for length in tokens_per_sample:
            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)

        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [original_seq_len, hidden_size]
        return patch_output, None


    def forward_debug(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        output = {}

        x = self.patch_embed(x)
        output["after_patch_embed"] = x.clone()

        batch_size = grid_thw.size(0)
        seq_len, hidden_dim = x.size()
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        class_embedding = self.class_embedding.view(1, -1)
        class_pos_emb = self.class_pos_emb.view(1, -1)
        class_tokens = class_embedding.expand(batch_size, -1)
        class_pos_embs = class_pos_emb.expand(batch_size, -1)

        tokens_per_sample = []

        for i in range(batch_size):
            t, h, w = grid_thw[i]
            tokens_per_sample.append((t * h * w).item())

        new_x = []
        start_idx = 0
        for i in range(batch_size):

            new_x.append(class_tokens[i:i+1])

            new_x.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        x = torch.cat(new_x, dim=0)
        new_rotary_pos_emb = []
        start_idx = 0
        for i in range(batch_size):
            new_rotary_pos_emb.append(class_pos_embs[i:i+1])
            new_rotary_pos_emb.append(rotary_pos_emb[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]

        rotary_pos_emb = torch.cat(new_rotary_pos_emb, dim=0)
        output["rotary_pos_emb"] = rotary_pos_emb.clone()
        output["class_embedding"] = self.class_embedding.clone()
        cu_seqlens = []
        cumulative_length = 0
        cu_seqlens.append(cumulative_length)  # Start from 0
        for length in tokens_per_sample:
            cumulative_length += int(length + 1)
            cu_seqlens.append(cumulative_length)


        cu_seqlens = torch.tensor(
            cu_seqlens,
            device=x.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32
        )

        x = x[:, None, :].contiguous()  # [s, h] -> [s, 1, h]

        x = self.pre_layernorm(x)
        output["after_pre_layernorm"] = x.clone()
        x = self.decoder(
            x,
            packed_seq_params=[PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ) for i in range(self.config.num_layers)],
            rotary_pos_emb=rotary_pos_emb.unsqueeze(1).unsqueeze(2),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask
        )
        x = x[:, 0, :].contiguous()  # [s, 1, h] -> [s, h]

        patch_output = []
        start_idx = 0
        for i in range(batch_size):
            start_idx += 1
            patch_output.append(x[start_idx:start_idx+tokens_per_sample[i]])
            start_idx += tokens_per_sample[i]
        patch_output = torch.cat(patch_output, dim=0)  # [original_seq_len, hidden_size]
        output["before_adapter"] = patch_output.clone()
        return output


class HybridVisionModel(RiceViTModel):
    """Rice-ViT backbone with optional SigLIP2 and DINOv3 feature fusion."""

    def __init__(
        self,
        config: VisionConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__(config=config, transformer_layer_spec=transformer_layer_spec, spatial_merge_size=spatial_merge_size)
        self.enable_siglip = enable_siglip
        self.enable_dinov3 = enable_dinov3
        self.siglip_model_path = siglip_model_path
        self.dinov3_model_path = dinov3_model_path
        self.freeze_external = freeze_external

        self.siglip_model = None
        self.dinov3_model = None

        # Get encoder hidden sizes from config or read from model configs
        siglip_hidden_size = self._get_encoder_hidden_size(
            siglip_model_path, config.siglip_hidden_size, "SigLIP2"
        )
        dinov3_hidden_size = self._get_encoder_hidden_size(
            dinov3_model_path, config.dinov3_hidden_size, "DINOv3"
        )

        # Use regular nn.Linear with known input dimensions instead of LazyLinear
        self.siglip_proj = (
            nn.Linear(siglip_hidden_size, config.hidden_size, bias=False)
            if enable_siglip else None
        )
        self.dinov3_proj = (
            nn.Linear(dinov3_hidden_size, config.hidden_size, bias=False)
            if enable_dinov3 else None
        )
        # Start from zero so initial behavior matches pure Rice-ViT.
        self.siglip_gate = nn.ParameterList([nn.Parameter(torch.tensor(0.0))]) if self.enable_siglip else None
        self.dinov3_gate = nn.ParameterList([nn.Parameter(torch.tensor(0.0))]) if self.enable_dinov3 else None

    def _get_encoder_hidden_size(
        self, model_path: str, config_hidden_size: int, model_name: str
    ) -> int:
        """Get encoder hidden size: prefer config value, fallback to reading model config."""
        # Use config-provided value if available (non-default)
        if config_hidden_size > 0:
            print_rank_0(
                f"[HybridVisionModel] Using {model_name} hidden_size={config_hidden_size} from config"
            )
            return config_hidden_size

        # Fallback: read from model config file
        if AutoConfig is None:
            print_rank_0(
                f"[HybridVisionModel] transformers unavailable; using default {model_name} hidden_size"
            )
            return 1152 if "siglip" in model_name.lower() else 1024

        try:
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # Try common hidden size attributes
            hidden_size = getattr(hf_config, "hidden_size", None)
            if hidden_size is None:
                # For vision models, try vision_config.hidden_size
                vision_config = getattr(hf_config, "vision_config", None)
                if vision_config is not None:
                    hidden_size = getattr(vision_config, "hidden_size", None)
            if hidden_size is not None:
                print_rank_0(
                    f"[HybridVisionModel] Read {model_name} hidden_size={hidden_size} from {model_path}"
                )
                return hidden_size
        except Exception as exc:
            print_rank_0(
                f"[HybridVisionModel] Failed to read {model_name} config from {model_path}: {exc}"
            )

        # Default fallback
        default_size = 1152 if "siglip" in model_name.lower() else 1024
        print_rank_0(
            f"[HybridVisionModel] Using default {model_name} hidden_size={default_size}"
        )
        return default_size

    def _is_dummy_pixels(self, pixel_values: Optional[torch.Tensor]) -> bool:
        if pixel_values is None:
            return True
        if not isinstance(pixel_values, torch.Tensor):
            return True
        if pixel_values.numel() <= 1:
            return True
        if pixel_values.ndim != 4:
            return True
        return False

    def _load_external_model(self, model_path: str, model_name: str):
        """Load external encoder model."""
        if AutoModel is None:
            print_rank_0(
                f"[HybridVisionModel] transformers AutoModel is unavailable; disabling {model_name}."
            )
            return None
        try:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            if self.freeze_external:
                model.requires_grad_(False)
                model.eval()
            print_rank_0(f"[HybridVisionModel] loaded {model_name} from {model_path}")
            return model
        except Exception as exc:
            print_rank_0(
                f"[HybridVisionModel] failed to load {model_name} from {model_path}: {exc}"
            )
            return None

    def load_external_models(self):
        """Load external encoder models. Call this after model initialization."""
        if self.enable_siglip and self.siglip_model is None:
            self.siglip_model = self._load_external_model(self.siglip_model_path, "SigLIP2")
        if self.enable_dinov3 and self.dinov3_model is None:
            self.dinov3_model = self._load_external_model(self.dinov3_model_path, "DINOv3")

    def _extract_sequence_features(self, model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        model = model.to(pixel_values.device)
        ctx = torch.no_grad() if self.freeze_external else contextlib.nullcontext()
        with ctx:
            outputs = model(pixel_values=pixel_values)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            features = outputs.last_hidden_state
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output.unsqueeze(1)
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            features = outputs[0]
        else:
            raise RuntimeError("Cannot parse external vision model outputs.")

        # For DINOv3: skip CLS token (1) and register tokens (4) to get pure patch features.
        # Reference: preprocessor.py dinov3_fwd()
        if hasattr(model.config, "num_register_tokens"):
            num_skip = 1 + model.config.num_register_tokens  # CLS + registers
            features = features[:, num_skip:, :]

        if features.ndim == 2:
            features = features.unsqueeze(1)
        return features

    def _interpolate_image_features(
        self,
        external_features: torch.Tensor,
        grid_thw: torch.Tensor,
        target_dtype: torch.dtype,
        target_device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Interpolate external encoder features to match Rice-ViT token count.

        Args:
            external_features: [B, external_tokens, hidden_size] from external encoder
            grid_thw: [B, 3] grid dimensions for each image
            target_dtype: target data type
            target_device: target device

        Returns:
            Interpolated features [total_target_tokens, hidden_size]
        """
        batch_size = external_features.size(0)
        image_token_lengths = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()
        if len(image_token_lengths) != batch_size:
            print_rank_0(
                "[HybridVisionModel] external image count "
                f"{batch_size} does not match grid_thw image count "
                f"{len(image_token_lengths)}; skipping this branch."
            )
            return None

        interpolated_list = []
        for i in range(batch_size):
            target_len = int(image_token_lengths[i])
            # [1, external_tokens, hidden_size] -> interpolate -> [1, target_len, hidden_size]
            feat = external_features[i:i + 1].permute(0, 2, 1)  # [1, hidden, external_tokens]
            feat_interpolated = torch.nn.functional.interpolate(
                feat,
                size=target_len,
                mode='linear',
                align_corners=False
            )  # [1, hidden, target_len]
            feat_interpolated = feat_interpolated.permute(0, 2, 1).squeeze(0)  # [target_len, hidden]
            interpolated_list.append(feat_interpolated)

        result = torch.cat(interpolated_list, dim=0)  # [total_target_tokens, hidden]
        return result.to(device=target_device, dtype=target_dtype)

    def _fuse_external_branch(
        self,
        base_embeddings: torch.Tensor,
        grid_thw: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        model: Optional[nn.Module],
        projector: Optional[nn.Module],
        gate: Optional[nn.Parameter],
    ) -> torch.Tensor:
        if self._is_dummy_pixels(pixel_values) or model is None or projector is None or gate is None:
            return base_embeddings

        seq_features = self._extract_sequence_features(model, pixel_values)
        # Project features: [B, external_tokens, external_hidden] -> [B, external_tokens, target_hidden]
        projected = projector(seq_features)
        # Interpolate to match Rice-ViT token count
        interpolated = self._interpolate_image_features(
            projected,
            grid_thw,
            target_dtype=base_embeddings.dtype,
            target_device=base_embeddings.device,
        )
        if interpolated is None:
            return base_embeddings
        return base_embeddings + torch.tanh(gate[0]) * interpolated

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        pixel_values_images_siglip: Optional[torch.Tensor] = None,
        pixel_values_images_dinov3: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        rice_embeddings, window_index = super().forward(x, grid_thw, **kwargs)
        if pixel_values_images_dinov3 is None:
            # Use SigLIP branch pixels as a fallback until a dedicated DINOv3 preprocessor stream is wired.
            pixel_values_images_dinov3 = pixel_values_images_siglip

        # Lazy load external models on first forward if not already loaded
        self.load_external_models()

        if self.enable_siglip:
            rice_embeddings = self._fuse_external_branch(
                base_embeddings=rice_embeddings,
                grid_thw=grid_thw,
                pixel_values=pixel_values_images_siglip,
                model=self.siglip_model,
                projector=self.siglip_proj,
                gate=self.siglip_gate,
            )

        if self.enable_dinov3:
            rice_embeddings = self._fuse_external_branch(
                base_embeddings=rice_embeddings,
                grid_thw=grid_thw,
                pixel_values=pixel_values_images_dinov3,
                model=self.dinov3_model,
                projector=self.dinov3_proj,
                gate=self.dinov3_gate,
            )

        return rice_embeddings, window_index
