import functools
import torch
import torch.nn as nn

import sys


sys.path.append("/workspace/Innovator-VL/aiak_megatron")


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
from aiak_training_llm.models.innovator_vl.innovator_vl_config import VisionConfig, get_vision_config
from aiak_training_llm.models.innovator_vl.innovator_vl_layer_spec import get_vision_layer_with_spec
from aiak_training_llm.models.innovator_vl.vision_model import RiceViTModel

import os
import torch
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)


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
    patch_size = 14
    size = 384 // patch_size * patch_size
    return CLIPViTModel(
        config,
        spec,
        add_class_token=False,
        class_token_len=0,
        img_h=size,
        img_w=size,
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
    for lyr in model.decoder.layers:
        disable_k_bias_for_layer(lyr)

    return model

initialize_distributed()
model = get_sliglip2_model()
print(model)


for param in model.parameters():
    param.requires_grad = False

model = get_dinov3_model()
print(model)


for param in model.parameters():
    param.requires_grad = False

spec = get_vit_layer_with_transformer_engine_spec()
vision_config = TransformerConfig(num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        kv_channels=64,
        normalization="LayerNorm",
        attention_dropout=0,
        hidden_dropout=0,
        layernorm_epsilon=1e-5,
        activation_func=torch.nn.functional.gelu,
        bias_activation_fusion=False,
        gated_linear_unit=False,
        num_query_groups=16,
        add_bias_linear=True,
        add_qkv_bias=True,
        position_embedding_type="rope",)
vision_config.patch_size = 14
vision_config.image_size = (1344, 1344)
vision_config.swiglu = False
vision_config.class_token_len = 0
vision_config.group_query_attention = False
vision_config.in_channels = 3
vision_config.spatial_merge_size = 2
model = RiceViTModel(vision_config, spec)
print(model)

for param in model.parameters():
    param.requires_grad = False

# torchrun --n_gpu_per_node 2 test_mcore_model.py