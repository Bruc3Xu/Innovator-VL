#!/usr/bin/env python3
"""
Randomly sample parquet files and copy to new directory
"""
import os
import random
import shutil
from pathlib import Path
from datetime import datetime

# Source and target directories
source_dir = "/mnt/innovator/data/chenshuang/RL_DATA/mm_science_vqa_rl_no_img/1229"
base_target_dir = "/root/innovator_data_wenzichen"

# Create new directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
target_dir = os.path.join(base_target_dir, f"mm_science_vqa_rl_no_img_{timestamp}")

# 获取allparquetfile
print(f"正在扫描源directory: {source_dir}")
parquet_files = list(Path(source_dir).glob("*.parquet"))
print(f"Found {len(parquet_files)} 个parquetfile")

if len(parquet_files) < 10:
    print(f"警告: 只有 {len(parquet_files)} 个file，少于10个")
    selected_files = parquet_files
else:
    # 随机选择10个file
    selected_files = random.sample(parquet_files, 10)

# 创建目标directory
os.makedirs(target_dir, exist_ok=True)
print(f"创建目标directory: {target_dir}")

# copy选in的file
print(f"\n开始copy {len(selected_files)} 个file...")
for i, file_path in enumerate(selected_files, 1):
    dest_path = os.path.join(target_dir, file_path.name)
    shutil.copy2(file_path, dest_path)
    print(f"[{i}/{len(selected_files)}] 已copy: {file_path.name}")

print(f"\n完成! file已保存to: {target_dir}")
print(f"\n选in的file列表:")
for file_path in selected_files:
    print(f"  - {file_path.name}")

