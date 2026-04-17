import torch
import torch.nn as nn
from typing import Optional


class HybridAdapter(nn.Module):
    """
    Hybrid adapter that manages 3 adapters for RiceViT, SigLIP2, and DINOv3 encoders,
    and implements feature fusion logic.

    The fusion strategy uses learnable gates to weighted combine features from
    all three encoders, with an optional final projection layer.
    """

    def __init__(
        self,
        ricevit_input_size: int,
        siglip_input_size: int,
        dinov3_input_size: int,
        output_size: int,
        spatial_merge_size: int = 2,
        fusion_type: str = "gated_sum",  # "gated_sum" or "concat"
        layer_norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.output_size = output_size
        self.spatial_merge_size = spatial_merge_size

        # RiceViT adapter: processes base RiceViT features
        self.ricevit_adapter = self._build_adapter(
            input_size=ricevit_input_size,
            output_size=output_size,
            spatial_merge_size=spatial_merge_size,
            layer_norm_eps=layer_norm_eps,
        )

        # SigLIP2 adapter: processes SigLIP2 features
        self.siglip_adapter = self._build_adapter(
            input_size=siglip_input_size,
            output_size=output_size,
            spatial_merge_size=1,  # External encoders don't use spatial merge
            layer_norm_eps=layer_norm_eps,
        )

        # DINOv3 adapter: processes DINOv3 features
        self.dinov3_adapter = self._build_adapter(
            input_size=dinov3_input_size,
            output_size=output_size,
            spatial_merge_size=1,  # External encoders don't use spatial merge
            layer_norm_eps=layer_norm_eps,
        )

        # Learnable fusion gates (initialized to 0 to start with RiceViT only)
        self.ricevit_gate = nn.Parameter(torch.tensor(1.0))  # Start with full RiceViT
        self.siglip_gate = nn.Parameter(torch.tensor(0.0))
        self.dinov3_gate = nn.Parameter(torch.tensor(0.0))

        # Optional fusion projection for concat mode
        if fusion_type == "concat":
            self.fusion_norm = nn.LayerNorm(output_size * 3, eps=layer_norm_eps)
            self.fusion_proj = nn.Linear(output_size * 3, output_size, bias=True)
        else:
            # For gated_sum, just use a layernorm on output
            self.fusion_norm = nn.LayerNorm(output_size, eps=layer_norm_eps)
            self.fusion_proj = None

    def _build_adapter(
        self,
        input_size: int,
        output_size: int,
        spatial_merge_size: int,
        layer_norm_eps: float,
    ) -> nn.Module:
        """Build a single adapter with LayerNorm + MLP structure like RicePatchMerger."""
        hidden_size = input_size * (spatial_merge_size ** 2) if spatial_merge_size > 1 else input_size
        return nn.Sequential(
            nn.LayerNorm(input_size, eps=layer_norm_eps),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
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
            ricevit_features: [B, N_rice, C_rice] or [total_tokens, C_rice] RiceViT features (base)
            siglip_features: [B, N_siglip, C_siglip] or [total_tokens, C_siglip] SigLIP2 features (optional)
            dinov3_features: [B, N_dino, C_dino] or [total_tokens, C_dino] DINOv3 features (optional)

        Returns:
            Fused features [B, N_out, C_out] or [total_tokens, C_out]
        """
        # Process each encoder's features through their respective adapter
        ricevit_out = self.ricevit_adapter(ricevit_features, window_index=window_index)

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
            if self.fusion_proj is not None:
                fused = self.fusion_proj(fused)
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