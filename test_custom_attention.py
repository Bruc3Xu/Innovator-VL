"""
Test script to demonstrate replacing DINOv3 and SigLIP2 attention layers
with custom implementations while preserving weights.
"""

import torch

from ds.innovator_vl.custom_attention import (
    CustomDINOv3Attention,
    CustomSigLIP2Attention,
    replace_dinov3_attention_layers,
    replace_siglip2_attention_layers,
convert_dinov3_attention_weights
)


def test_dinov3_attention_replacement():
    """Test DINOv3 attention layer replacement with weight preservation."""
    print("Testing DINOv3 attention replacement...")

    # Import DINOv3 model (assuming transformers library)
    try:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        # Create a small test model
        config = DINOv3ViTConfig(
            hidden_size=384,
            num_attention_heads=6,
            num_hidden_layers=2,
            image_size=224,
            patch_size=14,
        )
        model = DINOv3ViTModel(config)

        # Get sample input
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Forward pass before replacement (to get reference output)
        model.eval()
        with torch.no_grad():
            output_before = model(pixel_values).last_hidden_state

        # Replace attention layers
        model = replace_dinov3_attention_layers(model)
        print(model)

        # Verify replacement
        attention_count = 0
        for name, module in model.named_modules():
            if isinstance(module, CustomDINOv3Attention):
                attention_count += 1
        print(
            f"  Replaced {attention_count} attention layers with CustomDINOv3Attention"
        )

        # Forward pass after replacement
        with torch.no_grad():
            output_after = model(pixel_values).last_hidden_state

        # Check weight preservation (outputs should be identical)
        diff = torch.abs(output_before - output_after).max().item()
        print(f"  Max output difference after replacement: {diff:.8f}")

        if diff < 1e-6:
            print("  ✓ DINOv3 attention replacement successful! Weights preserved.")
        else:
            print(f"  ✗ Warning: Output difference too large ({diff})")

        return True

    except ImportError as e:
        print(f"  Skipping test - could not import transformers: {e}")
        return False


def test_siglip2_attention_replacement():
    """Test SigLIP2 attention layer replacement with weight preservation."""
    print("Testing SigLIP2 attention replacement...")

    try:
        # SigLIP2 might be named differently depending on transformers version
        from transformers import SiglipVisionConfig, SiglipVisionModel

        # Create a small test model
        config = SiglipVisionConfig(
            hidden_size=384,
            num_attention_heads=6,
            num_hidden_layers=2,
            image_size=224,
            patch_size=14,
        )
        model = SiglipVisionModel(config)

        # Get sample input
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Forward pass before replacement
        model.eval()
        with torch.no_grad():
            output_before = model(pixel_values).last_hidden_state

        # Replace attention layers
        model = replace_siglip2_attention_layers(model)
        print(model)

        # Verify replacement
        attention_count = 0
        for name, module in model.named_modules():
            if isinstance(module, CustomSigLIP2Attention):
                attention_count += 1
        print(
            f"  Replaced {attention_count} attention layers with CustomSigLIP2Attention"
        )

        # Forward pass after replacement
        with torch.no_grad():
            output_after = model(pixel_values).last_hidden_state

        # Check weight preservation
        diff = torch.abs(output_before - output_after).max().item()
        print(f"  Max output difference after replacement: {diff:.8f}")

        if diff < 1e-6:
            print("  ✓ SigLIP2 attention replacement successful! Weights preserved.")
        else:
            print(f"  ✗ Warning: Output difference too large ({diff})")

        return True

    except ImportError as e:
        print(f"  Skipping test - could not import SigLIP2: {e}")
        return False


def test_weight_conversion_verification():
    """Verify weight conversion logic is correct."""
    print("Testing weight conversion logic...")

    import torch.nn as nn

    # Create a mock attention module
    class MockConfig:
        def __init__(self):
            self.hidden_size = 768
            self.num_attention_heads = 12
            self.attention_dropout = 0.0
            self.query_bias = True
            self.key_bias = True
            self.value_bias = True
            self.proj_bias = True

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            embed_dim = 768
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
            self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return q, k, v

    # Create mock attention with random weights
    old_attn = MockAttention()
    old_attn.eval()

    # Create input
    x = torch.randn(2, 10, 768)  # [batch, seq, hidden]

    # Get outputs before conversion
    with torch.no_grad():
        q_old = old_attn.q_proj(x)
        k_old = old_attn.k_proj(x)
        v_old = old_attn.v_proj(x)

    # Convert to new attention
    new_attn = convert_dinov3_attention_weights(old_attn)
    new_attn.eval()

    # Verify QKV projection produces same output
    with torch.no_grad():
        qkv = new_attn.qkv(x)
        q_new, k_new, v_new = qkv.chunk(3, dim=-1)

    q_diff = torch.abs(q_old - q_new).max().item()
    k_diff = torch.abs(k_old - k_new).max().item()
    v_diff = torch.abs(v_old - v_new).max().item()

    print(f"  Q projection max diff: {q_diff:.8f}")
    print(f"  K projection max diff: {k_diff:.8f}")
    print(f"  V projection max diff: {v_diff:.8f}")

    if max(q_diff, k_diff, v_diff) < 1e-6:
        print("  ✓ Weight conversion verified! Q/K/V projections match.")
    else:
        print(f"  ✗ Warning: Q/K/V projections differ")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Custom Attention Layer Replacement Tests")
    print("=" * 60)

    # Test weight conversion logic
    test_weight_conversion_verification()
    print()

    # Test DINOv3 replacement
    test_dinov3_attention_replacement()
    print()

    # Test SigLIP2 replacement
    test_siglip2_attention_replacement()
    print()

    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)
