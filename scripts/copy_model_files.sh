# find /mnt/si00068187c7/default/innovator_vl/models/qwen3-8b-hybrid-vit-stage0 -type f -not -iname '*safetensors*' -exec cp {} checkpoints/cpt/ ';'

find /mnt/si00068187c7/default/innovator_vl/Innovator-VL/checkpoints/cpt -type f -not -iname '*safetensors*' -exec cp {} checkpoints/instruct_release ';'