"""
Custom Attention layers for DINOv3 and SigLIP2 models.
Converts separate Q/K/V projections to unified QKV projection while preserving weights.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class CustomDINOv3Attention(nn.Module):
    """
    Custom attention for DINOv3 that uses unified QKV projection.
    Compatible with RiceSdpaAttention style but adapted for DINOv3's interface.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # Unified QKV projection (for weight conversion)
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=config.query_bias)

        # Output projection
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, seq_len, _ = hidden_states.size()

        # Compute Q, K, V using unified projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        query_states, key_states, value_states = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # DINOv3 uses position embeddings without prefix tokens
            # The cos/sin from rope_embeddings match the patch tokens
            num_prefix_tokens = seq_len - cos.shape[0]
            if num_prefix_tokens > 0:
                # Apply RoPE only to non-prefix tokens
                q_prefix, q_patches = (
                    query_states[:, :, :num_prefix_tokens, :],
                    query_states[:, :, num_prefix_tokens:, :],
                )
                k_prefix, k_patches = (
                    key_states[:, :, :num_prefix_tokens, :],
                    key_states[:, :, num_prefix_tokens:, :],
                )
                q_patches, k_patches = apply_rotary_pos_emb_vision(
                    q_patches,
                    k_patches,
                    cos.unsqueeze(0).unsqueeze(0),
                    sin.unsqueeze(0).unsqueeze(0),
                )
                query_states = torch.cat([q_prefix, q_patches], dim=2)
                key_states = torch.cat([k_prefix, k_patches], dim=2)
            else:
                query_states, key_states = apply_rotary_pos_emb_vision(
                    query_states,
                    key_states,
                    cos.unsqueeze(0).unsqueeze(0),
                    sin.unsqueeze(0).unsqueeze(0),
                )

        # Use SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output, None


def convert_dinov3_attention_weights(old_attention):
    """
    Convert DINOv3ViTAttention weights to CustomDINOv3Attention format.

    Weight mapping:
    - q_proj.weight, k_proj.weight, v_proj.weight -> qkv.weight (concatenated)
    - q_proj.bias, k_proj.bias, v_proj.bias -> qkv.bias (concatenated)
    - o_proj.weight -> proj.weight
    - o_proj.bias -> proj.bias
    """
    new_attention = CustomDINOv3Attention(old_attention.config)

    with torch.no_grad():
        # Concatenate Q, K, V weights
        q_weight = old_attention.q_proj.weight
        k_weight = old_attention.k_proj.weight
        v_weight = old_attention.v_proj.weight
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        new_attention.qkv.weight.copy_(qkv_weight)

        # Concatenate Q, K, V biases if they exist
        if old_attention.q_proj.bias is not None:
            q_bias = old_attention.q_proj.bias
            k_bias = old_attention.k_proj.bias
            v_bias = old_attention.v_proj.bias
            if k_bias is None:
                k_bias = torch.zeros_like(q_bias)

            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            new_attention.qkv.bias.copy_(qkv_bias)

        # Copy output projection weights
        new_attention.proj.weight.copy_(old_attention.o_proj.weight)
        if old_attention.o_proj.bias is not None:
            new_attention.proj.bias.copy_(old_attention.o_proj.bias)

    return new_attention


def replace_dinov3_attention_layers(model):
    """Replace all attention layers in a DINOv3 model with custom implementation."""
    for name, module in model.named_modules():
        if module.__class__.__name__ == "DINOv3ViTAttention":
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            new_attention = convert_dinov3_attention_weights(module)
            setattr(parent, child_name, new_attention)

    return model


class CustomSigLIP2Attention(nn.Module):
    """
    Custom attention for SigLIP2 that uses unified QKV projection.
    Similar to RiceSdpaAttention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = getattr(config, "attention_dropout", 0.0)

        # Unified QKV projection
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)

        # Output projection
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, seq_len, _ = hidden_states.size()

        # Compute Q, K, V using unified projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query_states, key_states, value_states = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # SigLIP2 might use different position embedding handling
            query_states, key_states = apply_rotary_pos_emb_vision(
                query_states,
                key_states,
                cos.unsqueeze(0).unsqueeze(0) if cos.dim() == 2 else cos,
                sin.unsqueeze(0).unsqueeze(0) if sin.dim() == 2 else sin,
            )

        # Use SDPA
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.proj(attn_output)

        return attn_output, None


def convert_siglip2_attention_weights(old_attention):
    """
    Convert SigLIP2 attention weights to CustomSigLIP2Attention format.
    """
    new_attention = CustomSigLIP2Attention(old_attention.config)

    with torch.no_grad():
        # Concatenate Q, K, V weights
        q_weight = old_attention.q_proj.weight
        k_weight = old_attention.k_proj.weight
        v_weight = old_attention.v_proj.weight
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        new_attention.qkv.weight.copy_(qkv_weight)

        # Concatenate biases
        if (
            hasattr(old_attention.q_proj, "bias")
            and old_attention.q_proj.bias is not None
        ):
            q_bias = old_attention.q_proj.bias
            k_bias = old_attention.k_proj.bias
            v_bias = old_attention.v_proj.bias
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            new_attention.qkv.bias.copy_(qkv_bias)

        # Copy output projection
        new_attention.proj.weight.copy_(old_attention.out_proj.weight)
        if (
            hasattr(old_attention.out_proj, "bias")
            and old_attention.out_proj.bias is not None
        ):
            new_attention.proj.bias.copy_(old_attention.out_proj.bias)

    return new_attention


def replace_siglip2_attention_layers(model):
    """Replace all attention layers in a SigLIP2 model with custom implementation."""
    for name, module in model.named_modules():
        # SigLIP2 might have different class names depending on implementation
        if "Attention" in module.__class__.__name__ and hasattr(module, "q_proj"):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model

            new_attention = convert_siglip2_attention_weights(module)
            setattr(parent, child_name, new_attention)

    return model
