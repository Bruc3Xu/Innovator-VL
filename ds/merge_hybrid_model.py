"""Merge weights for Hybrid Vision Model with SigLIP2 and DINOv3 encoders."""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from transformers import (
     AutoConfig, AutoProcessor,
     Qwen2Tokenizer
)
from transformers import logging
from huggingface_hub import hf_hub_download, snapshot_download


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from innovator_vl.configuration_innovator_vl import InnovatorVlConfig, HybridVitConfig
from innovator_vl.modeling_innovator_vl import InnovatorVl_ForConditionalGeneration

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
CUDA_DEVICE = 0


def cosine_similarity(a, b):
    """Calculate cosine similarity between two tensors."""
    a, b = a.flatten().float(), b.flatten().float()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))


def create_test_image():
    """Create a test image."""
    img = Image.new('RGB', (560, 560), color='red')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 474, 474], fill='blue')
    draw.text((100, 100), "TEST", fill='white')
    return img


def load_empty_model(llm_path, enable_siglip=True, enable_dinov3=True):
    """Load an empty InnovatorVl model with HybridVitConfig."""
    print("Loading tokenizer and processor...")
    qwen_vl_path = "/mnt/si00068187c7/default/innovator_vl/models/Qwen2.5-VL-7B-Instruct"
    tokenizer = Qwen2Tokenizer.from_pretrained(
        qwen_vl_path,
        trust_remote_code=True,
        device_map={"": f"cuda:{CUDA_DEVICE}"}
    )
    processor = AutoProcessor.from_pretrained(qwen_vl_path)
    processor.image_processor.temporal_patch_size = 1
    processor.image_processor.max_pixels = 1600 * 1600

    # Create config with HybridVit
    llava_ov_config = InnovatorVlConfig()
    llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
    llava_ov_config.text_config.update(llm_config.to_dict())

    # Update vision config to use HybridVit
    llava_ov_config.vision_config = HybridVitConfig(
        text_hidden_size=llava_ov_config.text_config.hidden_size,
        freeze_external=True,
    )

    model = InnovatorVl_ForConditionalGeneration(llava_ov_config)
    return model, processor, tokenizer


def load_rice_vit_weights(model, vit_path):
    """Load RiceViT weights into the vision model."""
    print(f"Loading RiceViT weights from: {vit_path}")

    if os.path.exists(vit_path):
        cache_path = os.path.join(vit_path, "model.safetensors")
    else:
        cache_path = hf_hub_download(vit_path, "model.safetensors")

    vit_weights = load_file(cache_path)

    # Mapping for RiceViT keys
    VIT_KEYS_MAPPING = {
        "vision_model.": "model.visual.rice_vit.",
        "model.visual.embeddings.": "model.visual.rice_vit.",
        "model.visual.patch_embedding.": "model.visual.rice_vit.patch_embed.proj.",
        "model.visual.encoder.layers.": "model.visual.rice_vit.blocks.",
        "model.visual.pre_layrnorm": "model.visual.rice_vit.pre_layernorm",
        ".layer_norm": ".norm",
        ".self_attn.out_proj.": ".attn.proj.",
    }

    def merge_qkv_weights(state_dict, block_prefix):
        """Merge Q, K, V weights into QKV."""
        q_w = state_dict[f"{block_prefix}.self_attn.q_proj.weight"]
        k_w = state_dict[f"{block_prefix}.self_attn.k_proj.weight"]
        v_w = state_dict[f"{block_prefix}.self_attn.v_proj.weight"]
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{block_prefix}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{block_prefix}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{block_prefix}.self_attn.v_proj.bias"]
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

        return {
            f"{block_prefix}.attn.qkv.weight": qkv_weight,
            f"{block_prefix}.attn.qkv.bias": qkv_bias
        }

    def convert_state_dict(state_dict):
        """Convert RiceViT weights to model format."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            for old_key, new_key in VIT_KEYS_MAPPING.items():
                if old_key in key:
                    key = key.replace(old_key, new_key)
            new_state_dict[key] = value

        # Merge QKV weights
        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if (key.startswith("model.visual.rice_vit.blocks.") and
                "self_attn" in key and
                ("q_proj" in key or "k_proj" in key or "v_proj" in key)):
                block_index = key.split('.')[3]
                block_prefix = f"model.visual.rice_vit.blocks.{block_index}"
                if f"{block_prefix}.self_attn.q_proj.weight" in new_state_dict:
                    merge_res = merge_qkv_weights(new_state_dict, block_prefix)
                    new_state_dict2.update(merge_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2

    vit_weights = convert_state_dict(vit_weights)

    # Remove post_layernorm if exists (not used in model)
    vit_weights.pop("model.visual.rice_vit.post_layernorm.weight", None)
    vit_weights.pop("model.visual.rice_vit.post_layernorm.bias", None)

    # Load weights
    model_state_dict = model.state_dict()
    loaded_keys = 0
    vit_keys = len(vit_weights)

    for vit_key in vit_weights:
        if vit_key not in model_state_dict:
            logger.warning(f"ViT key {vit_key} not found in model, skipping...")
            continue
        model_state_dict[vit_key] = vit_weights[vit_key].clone()
        loaded_keys += 1

    assert loaded_keys == vit_keys, f"ViT weight loading incomplete: {loaded_keys}/{vit_keys}"
    model.load_state_dict(model_state_dict)
    print(f"RiceViT weights loaded: {loaded_keys}/{len(model_state_dict)} parameters")

    return vit_weights, loaded_keys

def load_siglip2_weights(model, siglip_path):
    """Load SigLIP2 weights into the model."""
    print(f"Loading SigLIP2 weights from: {siglip_path}")
    if os.path.exists(siglip_path):
        cache_path = os.path.join(siglip_path, "model.safetensors")
        if not os.path.exists(cache_path):
            cache_path = siglip_path  # Try as direct path
    else:
        cache_path = hf_hub_download(siglip_path, "model.safetensors")

    siglip_weights = load_file(cache_path)

    # Mapping for SigLIP2 keys
    SIGLIP_KEYS_MAPPING = {
        "vision_model.": "model.visual.siglip_model.",
        ".self_attn.out_proj.": ".attn.proj.",
    }

    def merge_qkv_weights(state_dict, block_prefix):
        """Merge Q, K, V weights into QKV."""
        q_w = state_dict[f"{block_prefix}.self_attn.q_proj.weight"]
        k_w = state_dict[f"{block_prefix}.self_attn.k_proj.weight"]
        v_w = state_dict[f"{block_prefix}.self_attn.v_proj.weight"]
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{block_prefix}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{block_prefix}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{block_prefix}.self_attn.v_proj.bias"]
        if k_b is None:
            k_b = torch.zeros_like(q_b)  # for dinov3 key bias = None
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

        return {
            f"{block_prefix}.self_attn.qkv.weight": qkv_weight,
            f"{block_prefix}.self_attn.qkv.bias": qkv_bias
        }

    def convert_state_dict(state_dict):
        """Convert SigLIP2 weights to model format."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if "text_model" in key:
                continue
            if key.endswith(".inv_freq"):
                continue
            if "head" in key:
                continue
            for old_key, replacement in SIGLIP_KEYS_MAPPING.items():
                if old_key in key:
                    key = key.replace(old_key, replacement)
            new_state_dict[key] = value

        # Merge QKV weights
        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if (key.startswith("model.visual.siglip_model.encoder.layers.") and
                "self_attn" in key and
                ("q_proj" in key or "k_proj" in key or "v_proj" in key)):
                block_index = key.split('.')[5]
                block_prefix = f"model.visual.siglip_model.encoder.layers.{block_index}"
                if f"{block_prefix}.self_attn.q_proj.weight" in new_state_dict:
                    merge_res = merge_qkv_weights(new_state_dict, block_prefix)
                    new_state_dict2.update(merge_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2

    siglip_weights = convert_state_dict(siglip_weights)

    siglip_weights.pop("model.visual.siglip_model.post_layernorm.weight", None)
    siglip_weights.pop("model.visual.siglip_model.post_layernorm.bias", None)
    siglip_weights.pop("logit_bias", None)
    siglip_weights.pop("logit_scale", None)

    print("siglip weights...")
    print(siglip_weights.keys())

    # Load weights
    model_state_dict = model.state_dict()
    loaded_keys = 0
    siglip_keys = len(siglip_weights)

    for key in siglip_weights:
        if key not in model_state_dict:
            logger.warning(f"SigLIP2 key {key} not found in model, skipping...")
            continue
        model_state_dict[key] = siglip_weights[key].clone()
        loaded_keys += 1

    model.load_state_dict(model_state_dict)
    print(f"SigLIP2 weights loaded: {loaded_keys}/{siglip_keys} parameters")

    return siglip_weights, loaded_keys


def load_dinov3_weights(model, dinov3_path):
    """Load DINOv3 weights into the model."""
    print(f"Loading DINOv3 weights from: {dinov3_path}")

    if os.path.exists(dinov3_path):
        cache_path = os.path.join(dinov3_path, "model.safetensors")
        if not os.path.exists(cache_path):
            # Try PyTorch format
            cache_path = os.path.join(dinov3_path, "pytorch_model.bin")
    else:
        try:
            cache_path = hf_hub_download(dinov3_path, "model.safetensors")
        except:
            cache_path = hf_hub_download(dinov3_path, "pytorch_model.bin")

    if cache_path.endswith('.safetensors'):
        dinov3_weights = load_file(cache_path)
    else:
        dinov3_weights = torch.load(cache_path, map_location="cpu")

    new_state_dict = {}
    for key, value in dinov3_weights.items():
        new_key = "DINOv3ViTModel." + key
        new_state_dict[new_key] = value
    dinov3_weights = new_state_dict

    # Mapping for DINOv3 keys
    DINOV3_KEYS_MAPPING = {
        "DINOv3ViTModel.": "model.visual.dinov3_model.",
        ".attention.o_proj.": ".attention.proj.",
    }

    def merge_qkv_weights(state_dict, block_prefix):
        """Merge Q, K, V weights into QKV."""
        q_w = state_dict[f"{block_prefix}.attention.q_proj.weight"]
        k_w = state_dict[f"{block_prefix}.attention.k_proj.weight"]
        v_w = state_dict[f"{block_prefix}.attention.v_proj.weight"]
        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = state_dict[f"{block_prefix}.attention.q_proj.bias"]
        k_b = state_dict.get(f"{block_prefix}.attention.k_proj.bias", None)
        v_b = state_dict[f"{block_prefix}.attention.v_proj.bias"]
        if k_b is None:
            k_b = torch.zeros_like(q_b)  # for dinov3 key bias = None
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

        return {
            f"{block_prefix}.attention.qkv.weight": qkv_weight,
            f"{block_prefix}.attention.qkv.bias": qkv_bias
        }


    def convert_state_dict(state_dict):
        """Convert DINOv3 weights to model format."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if "layer_scale" in key:
                continue
            for old_key, replacement in DINOV3_KEYS_MAPPING.items():
                if old_key in key:
                    key = key.replace(old_key, replacement)
            new_state_dict[key] = value
        # Merge QKV weights
        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if (key.startswith("model.visual.dinov3_model.layer.") and
                "attention" in key and
                ("q_proj" in key or "k_proj" in key or "v_proj" in key)):
                block_index = key.split('.')[4]
                block_prefix = f"model.visual.dinov3_model.layer.{block_index}"
                if f"{block_prefix}.attention.q_proj.weight" in new_state_dict:
                    merge_res = merge_qkv_weights(new_state_dict, block_prefix)
                    new_state_dict2.update(merge_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2

    dinov3_weights = convert_state_dict(dinov3_weights)
    dinov3_weights.pop("model.visual.dinov3_model.norm.weight", None)
    dinov3_weights.pop("model.visual.dinov3_model.norm.bias", None)

    print(dinov3_weights.keys())
    # Load weights
    model_state_dict = model.state_dict()
    loaded_keys = 0
    dinov3_keys = len(dinov3_weights)

    for key in dinov3_weights:
        if key not in model_state_dict:
            logger.warning(f"DINOv3 key {key} not found in model, skipping...")
            continue
        model_state_dict[key] = dinov3_weights[key].clone()
        loaded_keys += 1

    model.load_state_dict(model_state_dict)
    print(f"DINOv3 weights loaded: {loaded_keys}/{dinov3_keys} parameters")

    return dinov3_weights, loaded_keys


def load_adapter_weights(model, adapter_path, cur_len=0):
    """Load HybridAdapter (merger) weights into the model."""
    print(f"Loading Adapter/Merger weights from: {adapter_path}")

    if not adapter_path:
        print("No adapter path provided, skipping...")
        return {}, cur_len

    if adapter_path.endswith('.safetensors'):
        adapter_weights = load_file(adapter_path)
    else:
        adapter_weights = torch.load(adapter_path, map_location="cpu")
        if "state_dict" in adapter_weights:
            adapter_weights = adapter_weights["state_dict"]

    # Mapping for adapter keys
    ADAPTER_KEYS_MAPPING = {
        "model.mm_projector": "model.visual.merger",
        "adapter": "model.visual.merger",  # Hybrid adapter sub-modules
    }

    def convert_state_dict(state_dict):
        """Convert adapter weights to model format."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            new_key = key
            for old_key, replacement in ADAPTER_KEYS_MAPPING.items():
                if old_key in key:
                    new_key = key.replace(old_key, replacement)
            new_state_dict[new_key] = value
        return new_state_dict

    adapter_weights = convert_state_dict(adapter_weights)

    # Load weights
    model_state_dict = model.state_dict()
    loaded_keys = 0
    adapter_keys = len(adapter_weights)

    for key in adapter_weights:
        if key not in model_state_dict:
            logger.warning(f"Adapter key {key} not found in model, skipping...")
            continue
        model_state_dict[key] = adapter_weights[key].clone()
        loaded_keys += 1

    model.load_state_dict(model_state_dict)
    print(f"Adapter weights loaded: {loaded_keys}/{adapter_keys} parameters")

    return adapter_weights, cur_len + loaded_keys


def load_llm_weights(model, llm_path, cur_len=0):
    """Load LLM weights into the language model."""
    print(f"Loading LLM weights from: {llm_path}")

    if os.path.exists(llm_path):
        cache_path = llm_path
    else:
        cache_path = snapshot_download(llm_path, allow_patterns="*.safetensors")

    llm_weights = {}
    if os.path.isdir(cache_path):
        for filename in os.listdir(cache_path):
            if filename.endswith('.safetensors'):
                filepath = os.path.join(cache_path, filename)
                weights = load_file(filepath)
                llm_weights.update(weights)
    elif cache_path.endswith('.safetensors'):
        llm_weights = load_file(cache_path)
    else:
        llm_weights = torch.load(cache_path, map_location="cpu")
        if "state_dict" in llm_weights:
            llm_weights = llm_weights["state_dict"]

    # Mapping for LLM keys
    LLM_KEYS_MAPPING = {
        "model.": "model.language_model.",
    }

    def convert_state_dict(state_dict):
        """Convert LLM weights to model format."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            new_key = key
            for old_key, replacement in LLM_KEYS_MAPPING.items():
                if old_key in key:
                    new_key = key.replace(old_key, replacement)
            new_state_dict[new_key] = value
        return new_state_dict

    llm_weights = convert_state_dict(llm_weights)

    # Add lm_head if missing
    if 'lm_head.weight' not in llm_weights:
        llm_weights['lm_head.weight'] = llm_weights['model.language_model.embed_tokens.weight']

    # Load weights
    model_state_dict = model.state_dict()
    loaded_keys = 0
    llm_keys = len(llm_weights)

    for key in llm_weights:
        if key not in model_state_dict:
            logger.warning(f"LLM key {key} not found in model, skipping...")
            continue
        model_state_dict[key] = llm_weights[key].clone()
        loaded_keys += 1

    model.load_state_dict(model_state_dict)
    print(f"LLM weights loaded: {loaded_keys}/{llm_keys} parameters")

    return llm_weights, cur_len + loaded_keys


def validate_model_consistency(model, img_path, sample_text, tokenizer, processor):
    """Validate the merged model consistency."""
    print("\n=== Validating Model Consistency ===")

    # Create test image if needed
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}, creating test image...")
        test_img = create_test_image()
        test_img.save("/tmp/test_image.jpg")
        img_path = "/tmp/test_image.jpg"

    # Load and process image
    sample_image = Image.open(img_path).convert("RGB")
    sample_image = sample_image.resize((560, 560))

    # Test vision model
    print("\nTesting vision model...")
    try:
        from transformers import Qwen2VLImageProcessor
        image_processor = Qwen2VLImageProcessor()
        image_processor.temporal_patch_size = 1
        processed = image_processor(sample_image, return_tensors="pt")

        with torch.no_grad():
            image_grid_thw = torch.tensor([[1, 40, 40]], device=model.device, dtype=torch.long)
            output = model.visual(
                processed['pixel_values'].to(device=model.device, dtype=model.dtype),
                grid_thw=image_grid_thw
            )
        print(f"Vision model output shape: {output.shape}")
        print("✅ Vision model test passed")
    except Exception as e:
        print(f"❌ Vision model test failed: {e}")

    # Test text model
    print("\nTesting language model...")
    try:
        inputs = tokenizer(sample_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"Language model output logits shape: {outputs.logits.shape}")
        print("✅ Language model test passed")
    except Exception as e:
        print(f"❌ Language model test failed: {e}")


def save_merged_model(model, output_path, tokenizer, processor):
    """Save the merged model."""
    print(f"\nSaving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    tokenizer.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print("✅ Model saved successfully")


def main(args):
    """Main function to merge model weights."""
    # Load empty model with HybridVit config
    model, processor, tokenizer = load_empty_model(
        args.llm_path,
        enable_siglip=args.enable_siglip,
        enable_dinov3=args.enable_dinov3
    )
    model.to(dtype=torch.float32)

    pretrain_weights = {}
    cur_len = 0

    # 1. Load RiceViT weights
    if args.vit_path:
        vit_weights, cur_len = load_rice_vit_weights(model, args.vit_path)
        pretrain_weights.update(vit_weights)

    # 2. Load SigLIP2 weights
    if args.siglip_path and args.enable_siglip:
        siglip_weights, siglip_len = load_siglip2_weights(model, args.siglip_path)
        pretrain_weights.update(siglip_weights)
        cur_len += siglip_len

    # 3. Load DINOv3 weights
    if args.dinov3_path and args.enable_dinov3:
        dinov3_weights, dinov3_len = load_dinov3_weights(model, args.dinov3_path)
        pretrain_weights.update(dinov3_weights)
        cur_len += dinov3_len

    # 4. Load Adapter/Merger weights
    if args.adapter_path:
        adapter_weights, cur_len = load_adapter_weights(model, args.adapter_path, cur_len)
        pretrain_weights.update(adapter_weights)

    # 5. Load LLM weights
    llm_weights, cur_len = load_llm_weights(model, args.llm_path, cur_len)
    pretrain_weights.update(llm_weights)

    # Load all weights into model
    model.load_state_dict(pretrain_weights, strict=False)

    # Validate model
    validate_model_consistency(
        model,
        args.img_path,
        args.sample_text,
        tokenizer,
        processor
    )

    # Save merged model
    save_merged_model(
        model.to(dtype=torch.bfloat16),
        args.output_path,
        tokenizer,
        processor
    )

    print("\n✅ Model merging completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Hybrid Vision Model weights (RiceViT + SigLIP2 + DINOv3 + Adapter + LLM)"
    )

    # Model paths
    parser.add_argument("--vit_path", type=str, required=True,
                        help="Path to RiceViT model")
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Path to LLM model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save merged model")
    parser.add_argument("--siglip_path", type=str, default="",
                        help="Path to SigLIP2 model (optional)")
    parser.add_argument("--dinov3_path", type=str, default="",
                        help="Path to DINOv3 model (optional)")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Path to HybridAdapter weights (optional)")

    # Feature flags
    parser.add_argument("--enable_siglip", action="store_true", default=True,
                        help="Enable SigLIP2 encoder")
    parser.add_argument("--enable_dinov3", action="store_true", default=True,
                        help="Enable DINOv3 encoder")

    # Test data
    parser.add_argument("--img_path", type=str,
                        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        help="Path to test image")
    parser.add_argument("--sample_text", type=str, default="Hello, my dog is cute",
                        help="Sample text for testing")

    args = parser.parse_args()
    main(args)
