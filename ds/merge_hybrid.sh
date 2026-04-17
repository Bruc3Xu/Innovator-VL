python merge_model.py \
  --vit_path /mnt/si00068187c7/default/innovator_vl/models/rice-vit-large-patch14-560 \
  --enable_siglip \
  --siglip_path /mnt/si00068187c7/default/innovator_vl/models/siglip2-so400m-patch14-384 \
  --enable_dinov3 \
  --dinov3_path /mnt/si00068187c7/default/innovator_vl/models/dinov3-vitl16-pretrain-lvd1689m \
  --llm_path /mnt/si00068187c7/default/innovator_vl/models/Qwen3-8B-Base \
  --img_path /mnt/si00068187c7/default/innovator_vl/Innovator-VL/ds/demo.jpeg \
  --output_path /mnt/si00068187c7/default/innovator_vl/models/qwen3-8b-hybrid-vit-stage0