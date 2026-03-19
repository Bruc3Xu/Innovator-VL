#!/usr/bin/env python3
"""
合并目录下所有parquetfile到一个parquetfile
"""
import os
import glob
import io
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value
from PIL.Image import Image as ImageObject
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa

def merge_parquet_files(input_dir, output_file):
    """
    递归查找目录下所有parquetfile并合并
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出parquetfile路径
    """
    # 递归查找所有parquetfile
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True))
    
    if not parquet_files:
        print(f"❌ 在 {input_dir} 中未找到任何parquetfile")
        return
    
    print(f"📁 找到 {len(parquet_files)} 个parquetfile:")
    for f in parquet_files[:5]:
        print(f"   - {f}")
    if len(parquet_files) > 5:
        print(f"   ... 还有 {len(parquet_files) - 5} 个file")
    
    # 加载所有数据集
    print(f"\n📥 正在Loading dataset...")
    dataset_list = []
    for parquet_file in tqdm(parquet_files, desc="Loading"):
        try:
            # 首先尝试使用 datasets 库加载
            try:
                dataset = load_dataset("parquet", data_files=parquet_file)['train']
            except (Exception, TypeError, ValueError) as e1:
                # 如果failed，使用 pyarrow 读取并手动处理
                print(f"   ⚠️  datasets 加载failed，使用 pyarrow 手动处理: {str(e1)[:100]}")
                try:
                    # 使用 pyarrow 读取（可能需要特殊处理 PIL Image）
                    # 尝试直接读取，如果failed则使用其他方法
                    try:
                        table = pq.read_table(parquet_file)
                    except Exception as read_err:
                        print(f"   ⚠️  pyarrow 读取failed: {read_err}")
                        raise e1  # 重新抛出原始错误
                    
                    # 读取所有数据为 Python 列表
                    try:
                        data = table.to_pylist()
                    except Exception as pylist_err:
                        print(f"   ⚠️  转换为 Python 列表failed: {pylist_err}")
                        # 尝试逐行读取
                        data = []
                        for batch in table.to_batches():
                            data.extend(batch.to_pylist())
                    
                    # 转换 images 字段：将 PIL Image 转换为 bytes 格式
                    def convert_image_to_bytes(img):
                        """将单个图像转换为 bytes 格式"""
                        if isinstance(img, ImageObject):
                            img_bytes = io.BytesIO()
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                            img.save(img_bytes, format="PNG")
                            return {"bytes": img_bytes.getvalue(), "path": None}
                        elif isinstance(img, dict):
                            # 已经是字典格式
                            return img
                        elif img is None:
                            return None
                        elif hasattr(img, '__class__') and 'Image' in str(type(img)):
                            # 可能是其他 Image 类型，尝试转换
                            try:
                                img_bytes = io.BytesIO()
                                if hasattr(img, 'mode') and img.mode != "RGB":
                                    img = img.convert("RGB")
                                img.save(img_bytes, format="PNG")
                                return {"bytes": img_bytes.getvalue(), "path": None}
                            except Exception as conv_err:
                                print(f"   ⚠️  图像转换failed: {conv_err}")
                                return {"bytes": None, "path": None}
                        else:
                            # 未知格式，尝试检查
                            return {"bytes": None, "path": None}
                    
                    # 处理每个样本的 images 字段
                    if "images" in table.column_names:
                        for i, sample in enumerate(data):
                            try:
                                if "images" in sample and sample["images"] is not None:
                                    if isinstance(sample["images"], list):
                                        sample["images"] = [convert_image_to_bytes(img) for img in sample["images"]]
                                    else:
                                        sample["images"] = convert_image_to_bytes(sample["images"])
                            except Exception as img_err:
                                print(f"   ⚠️  处理样本 {i} 的 images 时出错: {img_err}")
                                # 继续处理其他样本
                    
                    # 使用 Dataset.from_dict 创建 Dataset
                    # Dataset.from_dict 需要 {column_name: [values]} 格式
                    if not data:
                        raise ValueError("数据为空，无法创建 Dataset")
                    
                    # 转换为字典格式：{column: [所有行的该列值]}
                    columns = list(data[0].keys())
                    dataset_dict = {}
                    for col in columns:
                        dataset_dict[col] = [row[col] for row in data]
                    
                    # 创建 Dataset
                    dataset = Dataset.from_dict(dataset_dict)
                    
                    # 使用 map 标准化 images 格式（参考 clean_data.py 的做法）
                    # 确保 images 字段统一为 {bytes, path} 格式
                    if "images" in dataset.column_names:
                        def standardize_row(example):
                            """标准化数据行格式，特别是 images 字段"""
                            # 处理 images 字段
                            if "images" in example and example["images"] is not None:
                                standardized_images = []
                                for img in example["images"]:
                                    if isinstance(img, dict) and "bytes" in img:
                                        # 已经是标准格式
                                        standardized_images.append({
                                            "bytes": img.get("bytes"),
                                            "path": str(img.get("path")) if img.get("path") is not None else None
                                        })
                                    elif isinstance(img, ImageObject):
                                        # PIL Image -> bytes
                                        img_bytes = io.BytesIO()
                                        if img.mode != "RGB":
                                            img = img.convert("RGB")
                                        img.save(img_bytes, format="PNG")
                                        standardized_images.append({
                                            "bytes": img_bytes.getvalue(),
                                            "path": None
                                        })
                                example["images"] = standardized_images
                            return example
                        
                        # 使用 map 标准化格式（不指定 features，让 datasets 自动推断）
                        dataset = dataset.map(
                            standardize_row,
                            desc=f"Standardizing {os.path.basename(parquet_file)}",
                            num_proc=1
                        )
                except Exception as e2:
                    print(f"   ⚠️  pyarrow 处理也failed: {e2}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 检查并转换 images 字段：将 PIL Image 转换为 bytes 格式（双重保险）
            if "images" in dataset.column_names:
                def convert_images_to_bytes(example):
                    """将 PIL Image 转换为 bytes 格式的字典"""
                    if "images" in example and example["images"] is not None:
                        converted_images = []
                        for img in example["images"]:
                            if isinstance(img, ImageObject):
                                # PIL Image -> bytes
                                img_bytes = io.BytesIO()
                                # 确保是 RGB 模式
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                img.save(img_bytes, format="PNG")
                                converted_images.append({"bytes": img_bytes.getvalue(), "path": None})
                            elif isinstance(img, dict):
                                # 已经是字典格式，保持不变
                                converted_images.append(img)
                            elif img is None:
                                # None 值，跳过
                                continue
                            else:
                                # 其他格式
                                converted_images.append({"bytes": None, "path": None})
                        example["images"] = converted_images
                    return example
                
                # 转换图像格式
                dataset = dataset.map(convert_images_to_bytes, desc=f"Converting images in {os.path.basename(parquet_file)}")
            
            dataset_list.append(dataset)
        except Exception as e:
            print(f"⚠️  加载 {parquet_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not dataset_list:
        print("❌ 没有success加载任何数据集")
        return
    
    print(f"\n✅ success加载 {len(dataset_list)} 个数据集")
    
    # 对齐列（处理可能的列不一致问题）
    print(f"\n🔧 对齐列结构...")
    all_features = set()
    for ds in dataset_list:
        all_features.update(ds.column_names)
    
    print(f"   发现 {len(all_features)} 个不同的列: {sorted(all_features)}")
    
    def add_missing_columns(example, current_columns):
        for feature in all_features:
            if feature not in current_columns:
                example[feature] = None
        return example
    
    dataset_list = [
        ds.map(lambda x: add_missing_columns(x, ds.column_names), desc="Aligning columns") 
        if set(ds.column_names) != all_features else ds 
        for ds in tqdm(dataset_list, desc="Aligning")
    ]
    
    # 统一Feature type（处理类型不匹配问题）
    if dataset_list:
        target_features = dataset_list[0].features
        new_list = []
        for ds in dataset_list:
            try:
                ds = ds.cast(target_features)
            except Exception as e:
                print(f"⚠️  转换Feature type时出错（继续处理）: {e}")
            new_list.append(ds)
        dataset_list = new_list
    
    # 合并所有数据集
    print(f"\n🔀 合并数据集...")
    merged_dataset = concatenate_datasets(dataset_list)
    
    print(f"\n📊 合并后的数据集统计:")
    print(f"   总Sample count: {len(merged_dataset)}")
    print(f"   列数: {len(merged_dataset.column_names)}")
    print(f"   列名: {merged_dataset.column_names}")
    
    # 保存到parquetfile
    print(f"\n💾 保存到 {output_file}...")
    merged_dataset.to_parquet(output_file)
    
    print(f"\n✅ 完成! 合并后的数据已保存到: {output_file}")
    print(f"   file大小: {os.path.getsize(output_file) / (1024**3):.2f} GB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python merge_parquet.py <输入目录> <输出file路径>")
        print("示例: python merge_parquet.py /path/to/input /path/to/output.parquet")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 如果输出路径没有 .parquet 扩展名，自动添加
    if not output_file.endswith('.parquet'):
        output_file = output_file + '.parquet'
        print(f"📝 自动添加 .parquet 扩展名: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    merge_parquet_files(input_dir, output_file)

