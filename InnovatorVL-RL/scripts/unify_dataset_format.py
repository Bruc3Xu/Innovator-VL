#!/usr/bin/env python3
"""
Unified dataset format script
Unify all dataset files to same format，确保每个字段的类型都一致
This avoids type mismatch issues when concatenating datasets
"""

import sys
import argparse
import os
import glob
from pathlib import Path
from typing import Dict, Any, Set, List
from collections import defaultdict

from datasets import load_dataset, Dataset, Features, Sequence, Image as DatasetImage, Value
from datasets.features.features import List as FeaturesList
from tqdm import tqdm
import io
from PIL import Image

# 需要移除的字段列表（训练时不需要的字段）
FIELDS_TO_REMOVE = [
    "reward_model",
    "vanilla_prompt",
]

# 标准字段类型定义
STANDARD_FEATURES = {
    "id": Value("string"),
    "problem": Value("string"),
    "answer": FeaturesList(Value("string")),
    "images": Sequence(DatasetImage(decode=True)),  # 统一使用 Image 类型
    "problem_type": Value("string"),
    "answer_type": Value("string"),
    "source": Value("string"),
    "prompt_type": Value("string"),
    "messages": Value("string"),
    "data_source": Value("string"),  # 允许 null
    "solution": Value("string"),  # 允许 null
    "avg_reward": Value("float"),  # 允许 null
    "options": Value("string"),  # 允许 null
    "tokens": Value("int64"),  # 允许 null
    "ability": Value("string"),  # 允许 null
    "format_guidance": Value("string"),  # 允许 null
}

def analyze_all_datasets(input_paths: List[str]) -> Dict[str, Any]:
    """
    分析all数据集，收集All fieldsand它们的类型
    返回：字段名 -> all出现过的类型的集合
    """
    print("=" * 80)
    print("步骤 1: 分析all数据集，收集字段信息...")
    print("=" * 80)
    
    all_fields: Set[str] = set()
    field_types: Dict[str, Set[str]] = defaultdict(set)
    
    # 展开all路径
    expanded_paths = []
    for path in input_paths:
        path = path.strip()
        if "*" in path:
            expanded_paths.extend(sorted(glob.glob(path)))
        elif os.path.isdir(path):
            expanded_paths.extend(sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)))
        elif path.endswith((".parquet", ".json")):
            expanded_paths.append(path)
    
    print(f"Found {len(expanded_paths)} 个数据集file")
    
    for path in tqdm(expanded_paths, desc="分析数据集"):
        try:
            if path.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=path)['train']
            elif path.endswith(".json"):
                dataset = load_dataset("json", data_files=path)['train']
            else:
                continue
            
            # 收集字段信息
            for field_name in dataset.column_names:
                if field_name in FIELDS_TO_REMOVE:
                    continue
                all_fields.add(field_name)
                field_type_str = str(dataset.features[field_name])
                field_types[field_name].add(field_type_str)
        except Exception as e:
            print(f"⚠️  跳过file {path}: {e}")
            continue
    
    print(f"\n发现 {len(all_fields)} 个唯一字段:")
    for field in sorted(all_fields):
        types = field_types[field]
        print(f"  - {field}: {len(types)} 种类型")
        if len(types) > 1:
            for t in sorted(types):
                print(f"      {t}")
    
    return {
        "all_fields": all_fields,
        "field_types": field_types,
        "expanded_paths": expanded_paths
    }

def determine_standard_features(all_fields: Set[str], field_types: Dict[str, Set[str]]) -> Features:
    """
    根据分析结果确定标准Feature type
    """
    print("\n" + "=" * 80)
    print("步骤 2: 确定标准Feature type...")
    print("=" * 80)
    
    standard_features_dict = {}
    
    for field in sorted(all_fields):
        if field in FIELDS_TO_REMOVE:
            continue
        
        # e.g.果字段在标准定义in，使用标准定义
        if field in STANDARD_FEATURES:
            standard_features_dict[field] = STANDARD_FEATURES[field]
            print(f"  ✅ {field}: 使用预定义标准类型")
        else:
            # 根据出现的类型推断标准类型
            types = field_types[field]
            type_strs = list(types)
            
            # 简单的类型推断逻辑
            if any("string" in t.lower() for t in type_strs):
                standard_features_dict[field] = Value("string")
                print(f"  📝 {field}: 推断as Value('string')")
            elif any("int" in t.lower() for t in type_strs):
                standard_features_dict[field] = Value("int64")
                print(f"  📝 {field}: 推断as Value('int64')")
            elif any("float" in t.lower() for t in type_strs):
                standard_features_dict[field] = Value("float")
                print(f"  📝 {field}: 推断as Value('float')")
            elif any("bool" in t.lower() for t in type_strs):
                standard_features_dict[field] = Value("bool")
                print(f"  📝 {field}: 推断as Value('bool')")
            else:
                # 默认使用 string
                standard_features_dict[field] = Value("string")
                print(f"  ⚠️  {field}: 无法推断，默认使用 Value('string')")
    
    return Features(standard_features_dict)

def convert_images_to_standard_format(images: Any) -> List[Any]:
    """
    将图像转换as标准format（PIL Image 对象，用于 Sequence(Image(...))）
    """
    if images is None or len(images) == 0:
        return []
    
    converted = []
    for img in images:
        if isinstance(img, Image.Image):
            # 已经是 PIL Image，直接使用
            converted.append(img)
        elif isinstance(img, dict):
            # 从 dict format加载
            if "bytes" in img and img["bytes"] is not None:
                try:
                    pil_img = Image.open(io.BytesIO(img["bytes"]))
                    if pil_img.mode in ("CMYK", "YCbCr", "LAB", "HSV", "P"):
                        pil_img = pil_img.convert("RGB")
                    converted.append(pil_img)
                except Exception as e:
                    print(f"⚠️  无法加载图像 bytes: {e}")
                    continue
            elif "path" in img and img["path"] is not None:
                try:
                    pil_img = Image.open(img["path"])
                    if pil_img.mode in ("CMYK", "YCbCr", "LAB", "HSV", "P"):
                        pil_img = pil_img.convert("RGB")
                    converted.append(pil_img)
                except Exception as e:
                    print(f"⚠️  无法加载图像路径 {img['path']}: {e}")
                    continue
        elif isinstance(img, str):
            # file路径
            try:
                pil_img = Image.open(img)
                if pil_img.mode in ("CMYK", "YCbCr", "LAB", "HSV", "P"):
                    pil_img = pil_img.convert("RGB")
                converted.append(pil_img)
            except Exception as e:
                print(f"⚠️  无法加载图像路径 {img}: {e}")
                continue
        else:
            print(f"⚠️  未知的图像format: {type(img)}")
    
    return converted

def normalize_sample(sample: Dict[str, Any], standard_features: Features) -> Dict[str, Any]:
    """
    将单个样本转换as标准format
    """
    normalized = {}
    
    for field_name, field_feature in standard_features.items():
        if field_name not in sample:
            # 字段不存在，设置as None（e.g.果允许）或默认值
            if isinstance(field_feature, Value) and field_feature.dtype == "string":
                normalized[field_name] = None
            else:
                normalized[field_name] = None
        else:
            value = sample[field_name]
            
            # 特殊处理 images 字段
            if field_name == "images":
                normalized[field_name] = convert_images_to_standard_format(value)
            # 特殊处理其他字段的类型转换
            elif isinstance(field_feature, Value):
                # 简单类型转换
                if value is None:
                    normalized[field_name] = None
                elif field_feature.dtype == "string":
                    normalized[field_name] = str(value) if value is not None else None
                elif field_feature.dtype == "int64":
                    try:
                        normalized[field_name] = int(value) if value is not None else None
                    except (ValueError, TypeError):
                        normalized[field_name] = None
                elif field_feature.dtype == "float":
                    try:
                        normalized[field_name] = float(value) if value is not None else None
                    except (ValueError, TypeError):
                        normalized[field_name] = None
                else:
                    normalized[field_name] = value
            elif isinstance(field_feature, FeaturesList):
                # List 类型
                if value is None:
                    normalized[field_name] = []
                elif isinstance(value, list):
                    # 转换列表in的每个元素
                    inner_feature = field_feature.feature
                    if isinstance(inner_feature, Value):
                        if inner_feature.dtype == "string":
                            normalized[field_name] = [str(v) if v is not None else "" for v in value]
                        else:
                            normalized[field_name] = value
                    else:
                        normalized[field_name] = value
                else:
                    normalized[field_name] = [value] if value is not None else []
            else:
                # 其他类型，保持原样
                normalized[field_name] = value
    
    return normalized

def process_dataset_file(input_path: str, standard_features: Features, output_path: str = None) -> Dataset:
    """
    处理单个数据集file，转换as标准format
    """
    print(f"\n处理file: {input_path}")
    
    # Loading dataset
    if input_path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=input_path)['train']
    elif input_path.endswith(".json"):
        dataset = load_dataset("json", data_files=input_path)['train']
    else:
        raise ValueError(f"不支持的fileformat: {input_path}")
    
    # 移除不需要的字段
    columns_to_keep = [col for col in dataset.column_names if col not in FIELDS_TO_REMOVE]
    if len(columns_to_keep) < len(dataset.column_names):
        dataset = dataset.select_columns(columns_to_keep)
        print(f"  移除了 {len(dataset.column_names) - len(columns_to_keep)} 个不需要的字段")
    
    # 转换每个样本
    print(f"  转换 {len(dataset)} 个样本...")
    
    def normalize_batch(examples):
        """批量归一化"""
        batch_size = len(examples[list(examples.keys())[0]])
        normalized_batch = {field: [] for field in standard_features.keys()}
        
        for i in range(batch_size):
            sample = {key: examples[key][i] for key in examples.keys()}
            normalized = normalize_sample(sample, standard_features)
            for field in standard_features.keys():
                normalized_batch[field].append(normalized.get(field))
        
        return normalized_batch
    
    # 确定要移除的列（不在标准特征in的列）
    columns_to_remove = [col for col in dataset.column_names if col not in standard_features]
    
    # 使用 map 批量处理
    dataset = dataset.map(
        normalize_batch,
        batched=True,
        batch_size=1000,
        desc="归一化样本",
        remove_columns=columns_to_remove if columns_to_remove else None
    )
    
    # 设置标准特征
    try:
        dataset = dataset.cast(standard_features)
        print(f"  ✅ success转换as标准format")
    except Exception as e:
        print(f"  ⚠️  cast failed: {e}")
        print(f"  使用 from_dict 重new创建...")
        # 重new创建数据集
        data_dict = {col: [dataset[j][col] for j in range(len(dataset))] 
                     for col in dataset.column_names}
        dataset = Dataset.from_dict(data_dict, features=standard_features)
        print(f"  ✅ 通过 from_dict success创建标准format数据集")
    
    # 保存（e.g.果指定了输出路径）
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        dataset.to_parquet(output_path)
        print(f"  💾 已保存to: {output_path}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="统一数据集format")
    parser.add_argument("input_paths", nargs="+", help="输入数据集路径（支持 glob 模式）")
    parser.add_argument("--output_dir", type=str, help="输出directory（e.g.果不指定，则覆盖原file）")
    parser.add_argument("--output_suffix", type=str, default="_unified", help="输出file后缀")
    parser.add_argument("--dry-run", action="store_true", help="只分析，不实际转换")
    
    args = parser.parse_args()
    
    # 步骤 1: 分析all数据集
    analysis_result = analyze_all_datasets(args.input_paths)
    all_fields = analysis_result["all_fields"]
    field_types = analysis_result["field_types"]
    expanded_paths = analysis_result["expanded_paths"]
    
    if args.dry_run:
        print("\n🔍 这是 dry-run 模式，不会实际转换file")
        return
    
    # 步骤 2: 确定标准特征
    standard_features = determine_standard_features(all_fields, field_types)
    
    # 步骤 3: 处理每个file
    print("\n" + "=" * 80)
    print("步骤 3: 转换all数据集file...")
    print("=" * 80)
    
    for input_path in tqdm(expanded_paths, desc="处理file"):
        try:
            if args.output_dir:
                # 保存to指定directory
                input_name = Path(input_path).stem
                output_path = os.path.join(args.output_dir, f"{input_name}{args.output_suffix}.parquet")
            else:
                # 覆盖原file（添加后缀）
                output_path = str(Path(input_path).with_suffix("")) + args.output_suffix + ".parquet"
            
            process_dataset_file(input_path, standard_features, output_path)
        except Exception as e:
            print(f"❌ 处理file {input_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ all数据集已统一format！")
    print("=" * 80)

if __name__ == "__main__":
    main()

