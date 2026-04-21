import os

from safetensors.torch import load_file


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(
            path, f"model-{i:05d}-of-{num_checkpoints:05d}.safetensors"
        )
        current_chunk = load_file(checkpoint_path)
        state_dict.update(current_chunk)
    return state_dict


def load_huggingface_checkpoint(load_path):
    """load ckpt"""
    state_dict = {}
    sub_dirs = [x for x in os.listdir(load_path) if x.endswith("safetensors")]
    if len(sub_dirs) == 1:
        checkpoint_name = "model.safetensors"
        state_dict = load_file(os.path.join(load_path, checkpoint_name), device="cpu")
    else:
        num_checkpoints = len(sub_dirs)
        state_dict = merge_transformers_sharded_states(load_path, num_checkpoints)
    return state_dict


# ckpt = load_file("dinov3-vitl16-pretrain-lvd1689m/model.safetensors", device="cpu")
# for k in ckpt.keys():
#     if "layer" in k:
#         continue
#     print(k)


# ckpt = load_file("siglip2-so400m-patch14-384/model.safetensors", device="cpu")
# for k in ckpt.keys():
#     if "layer" in k:
#         continue
#     print(k)


ckpt = load_huggingface_checkpoint(
    "/mnt/si00068187c7/default/innovator_vl/models/qwen3-8b-hybrid-vit-stage0"
)
for k in ckpt.keys():
    if "merger" in k:
        print(k)
