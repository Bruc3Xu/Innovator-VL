""" Adapters """

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Optional
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module


@dataclass
class AdapterSubmodules:
    """Adapter sub-modules."""
    layernorm: Union[ModuleSpec, type] = None
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


@dataclass
class HybridAdapterSubmodules:
    """HybridAdapter sub-modules for 3 encoders."""
    # RiceViT adapter submodules
    ricevit_layernorm: Union[ModuleSpec, type] = None
    ricevit_fc1: Union[ModuleSpec, type] = None
    ricevit_fc2: Union[ModuleSpec, type] = None
    # SigLIP2 adapter submodules
    siglip_layernorm: Union[ModuleSpec, type] = None
    siglip_fc1: Union[ModuleSpec, type] = None
    siglip_fc2: Union[ModuleSpec, type] = None
    # DINOv3 adapter submodules
    dinov3_layernorm: Union[ModuleSpec, type] = None
    dinov3_fc1: Union[ModuleSpec, type] = None
    dinov3_fc2: Union[ModuleSpec, type] = None
    # Fusion projection submodules
    fusion_layernorm: Union[ModuleSpec, type] = None
    fusion_fc: Union[ModuleSpec, type] = None


class Adapter(MegatronModule):
    """ Adaptor  """
    def __init__(self, 
        config: TransformerConfig,
        submodules: AdapterSubmodules,
        input_size: int,
        output_size: int,
        spatial_merge_size: int = 2
    ) -> None:
        super().__init__(config=config)
        self.hidden_size = input_size * (spatial_merge_size**2)

        self.layernorm = build_module(
            submodules.layernorm,
            config=config,
            hidden_size=input_size,
            eps=config.layernorm_epsilon,
        )

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )
        
        self.activation_func = config.activation_func

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.hidden_size,
            output_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            parallel_mode=None,
            skip_weight_param_allocation=False,
        )

    def forward(self, x: torch.Tensor, window_index: torch.LongTensor = None) -> torch.Tensor:
        """ Forward pass."""
        x = self.layernorm(x).view(-1, self.hidden_size)
        x, _ = self.linear_fc1(x)
        x = self.activation_func(x)
        x, _ = self.linear_fc2(x)
        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            x = x[reverse_indices, :].contiguous()
        return x


class HybridAdapter(MegatronModule):
    """
    Hybrid adapter that manages 3 adapters for RiceViT, SigLIP2, and DINOv3 encoders,
    and implements feature fusion logic.

    The fusion strategy uses learnable gates to weighted combine features from
    all three encoders, with an optional final projection layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridAdapterSubmodules,
        ricevit_input_size: int,
        siglip_input_size: int,
        dinov3_input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
        fusion_type: str = "gated_sum",  # "gated_sum" or "concat"
    ) -> None:
        super().__init__(config=config)
        self.fusion_type = fusion_type
        self.output_size = output_size
        self.spatial_merge_size = spatial_merge_size

        # RiceViT adapter: processes base RiceViT features
        self.ricevit_adapter = self._build_adapter(
            config=config,
            layernorm_spec=submodules.ricevit_layernorm,
            fc1_spec=submodules.ricevit_fc1,
            fc2_spec=submodules.ricevit_fc2,
            input_size=ricevit_input_size,
            output_size=output_size,
            spatial_merge_size=spatial_merge_size,
        )

        # SigLIP2 adapter: processes SigLIP2 features
        self.siglip_adapter = self._build_adapter(
            config=config,
            layernorm_spec=submodules.siglip_layernorm,
            fc1_spec=submodules.siglip_fc1,
            fc2_spec=submodules.siglip_fc2,
            input_size=siglip_input_size,
            output_size=output_size,
            spatial_merge_size=1,  # External encoders don't use spatial merge
        )

        # DINOv3 adapter: processes DINOv3 features
        self.dinov3_adapter = self._build_adapter(
            config=config,
            layernorm_spec=submodules.dinov3_layernorm,
            fc1_spec=submodules.dinov3_fc1,
            fc2_spec=submodules.dinov3_fc2,
            input_size=dinov3_input_size,
            output_size=output_size,
            spatial_merge_size=1,  # External encoders don't use spatial merge
        )

        # Learnable fusion gates (initialized to 0 to start with RiceViT only)
        self.ricevit_gate = nn.Parameter(torch.tensor(1.0))  # Start with full RiceViT
        self.siglip_gate = nn.Parameter(torch.tensor(0.0))
        self.dinov3_gate = nn.Parameter(torch.tensor(0.0))

        # Optional fusion projection for concat mode
        if fusion_type == "concat":
            self.fusion_norm = build_module(
                submodules.fusion_layernorm,
                config=config,
                hidden_size=output_size * 3,
                eps=config.layernorm_epsilon,
            )
            self.fusion_proj = build_module(
                submodules.fusion_fc,
                output_size * 3,
                output_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                parallel_mode=None,
                skip_weight_param_allocation=False,
            )
        else:
            # For gated_sum, just use a layernorm on output
            self.fusion_norm = build_module(
                submodules.fusion_layernorm,
                config=config,
                hidden_size=output_size,
                eps=config.layernorm_epsilon,
            )
            self.fusion_proj = None

    def _build_adapter(
        self,
        config: TransformerConfig,
        layernorm_spec,
        fc1_spec,
        fc2_spec,
        input_size: int,
        output_size: int,
        spatial_merge_size: int,
    ) -> Adapter:
        """Build a single adapter with given specs."""
        adapter_submodules = AdapterSubmodules(
            layernorm=layernorm_spec,
            linear_fc1=fc1_spec,
            linear_fc2=fc2_spec,
        )
        return Adapter(
            config=config,
            submodules=adapter_submodules,
            input_size=input_size,
            output_size=output_size,
            spatial_merge_size=spatial_merge_size,
        )

    def forward(
        self,
        ricevit_features: torch.Tensor,
        siglip_features: Optional[torch.Tensor] = None,
        dinov3_features: Optional[torch.Tensor] = None,
        window_index: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass that fuses features from 3 encoders.

        Args:
            ricevit_features: [B, N_rice, C_rice] RiceViT features (base)
            siglip_features: [B, N_siglip, C_siglip] SigLIP2 features (optional)
            dinov3_features: [B, N_dino, C_dino] DINOv3 features (optional)
            window_index: Optional window index for reordering

        Returns:
            Fused features [B, N_out, C_out]
        """
        # Process each encoder's features through their respective adapter
        ricevit_out = self.ricevit_adapter(ricevit_features, window_index)

        # Process external encoder features if provided
        if siglip_features is not None:
            siglip_out = self.siglip_adapter(siglip_features)
            # Interpolate to match RiceViT token count if needed
            if siglip_out.size(0) != ricevit_out.size(0):
                siglip_out = self._interpolate_features(siglip_out, ricevit_out.size(0))
        else:
            siglip_out = torch.zeros_like(ricevit_out)

        if dinov3_features is not None:
            dinov3_out = self.dinov3_adapter(dinov3_features)
            # Interpolate to match RiceViT token count if needed
            if dinov3_out.size(0) != ricevit_out.size(0):
                dinov3_out = self._interpolate_features(dinov3_out, ricevit_out.size(0))
        else:
            dinov3_out = torch.zeros_like(ricevit_out)

        # Fuse features
        if self.fusion_type == "gated_sum":
            # Gated weighted sum
            fused = (
                torch.sigmoid(self.ricevit_gate) * ricevit_out +
                torch.sigmoid(self.siglip_gate) * siglip_out +
                torch.sigmoid(self.dinov3_gate) * dinov3_out
            )
        elif self.fusion_type == "concat":
            # Concatenate and project
            fused = torch.cat([ricevit_out, siglip_out, dinov3_out], dim=-1)
            fused = self.fusion_norm(fused)
            fused, _ = self.fusion_proj(fused)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Final normalization
        fused = self.fusion_norm(fused)

        return fused

    def _interpolate_features(
        self,
        features: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Interpolate features to match target sequence length.

        Args:
            features: [B, N, C] or [total_tokens, C] features
            target_length: Target sequence length

        Returns:
            Interpolated features
        """
        original_shape = features.shape

        if features.ndim == 2:
            # [total_tokens, C] -> [1, C, total_tokens]
            features = features.unsqueeze(0).permute(0, 2, 1)
        elif features.ndim == 3:
            # [B, N, C] -> [B, C, N]
            features = features.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected feature shape: {original_shape}")

        # Interpolate
        features = torch.nn.functional.interpolate(
            features,
            size=target_length,
            mode='linear',
            align_corners=False
        )

        # Restore shape
        if len(original_shape) == 2:
            # [1, C, target_length] -> [target_length, C]
            features = features.permute(0, 2, 1).squeeze(0)
        else:
            # [B, C, target_length] -> [B, target_length, C]
            features = features.permute(0, 2, 1)

        return features.contiguous()