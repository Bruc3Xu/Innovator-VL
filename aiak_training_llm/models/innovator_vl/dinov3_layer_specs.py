# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""DINOv3 ViT layer specs"""

import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    LNImpl = FusedLayerNorm
except Exception:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm


def apply_rotary_pos_emb_2d_dinov3(
    t, freqs, config, cu_seqlens=None, num_prefix_tokens=5
):
    """Apply 2D Rotary Position Embedding for DINOv3.

    DINOv3 applies RoPE only to patch tokens, not to prefix tokens (CLS, registers).
    This function handles the special case where we need to skip prefix tokens.

    Args:
        t: Input tensor of shape (seq_len, batch, num_heads, head_dim).
        freqs: Precomputed frequency tensor from RoPE of shape (seq_len, 1, 1, head_dim).
        config: Transformer configuration.
        cu_seqlens: Cumulative sequence lengths for packed sequences.
        num_prefix_tokens: Number of prefix tokens to skip (CLS + registers).

    Returns:
        Tensor with RoPE applied to non-prefix tokens only.
    """
    if cu_seqlens is not None:
        # Packed sequence case - not commonly used for vision models
        return _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)

    seq_len = t.shape[0]

    if num_prefix_tokens == 0 or seq_len <= num_prefix_tokens:
        # No prefix tokens or all tokens are prefix tokens
        return _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)

    # Split prefix and patch tokens
    t_prefix = t[:num_prefix_tokens]
    t_patches = t[num_prefix_tokens:]

    # For patch tokens, use freqs from index 0 onwards
    freqs_patches = freqs[: t_patches.shape[0]]

    # Apply RoPE only to patch tokens
    t_patches_rotated = _apply_rotary_pos_emb_bshd(
        t_patches, freqs_patches, rotary_interleaved=config.rotary_interleaved
    )

    # Concatenate prefix and rotated patch tokens
    return torch.cat([t_prefix, t_patches_rotated], dim=0)


def get_dinov3_layer_with_transformer_engine_spec(num_prefix_tokens=5):
    """Returns DINOv3 ViT layer spec with Transformer Engine layers.

    Args:
        num_prefix_tokens: Number of prefix tokens (CLS + registers).

    Returns:
        ModuleSpec: Layer specification for DINOv3.
    """
    mlp = _get_dinov3_mlp_module_spec(use_te=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,  # DINOv3 uses pre-LN in attention TE layer
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_2d_dinov3,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,  # DINOv3 uses pre-LN in MLP TE layer
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_dinov3_layer_with_local_spec(num_prefix_tokens=5):
    """Returns DINOv3 ViT layer spec with MCore local layers.

    Args:
        num_prefix_tokens: Number of prefix tokens (CLS + registers).

    Returns:
        ModuleSpec: Layer specification for DINOv3.
    """
    mlp = _get_dinov3_mlp_module_spec(use_te=False)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    apply_rotary_fn=apply_rotary_pos_emb_2d_dinov3,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def _get_dinov3_mlp_module_spec(use_te=True):
    """Get MLP module spec for DINOv3."""
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
    )
