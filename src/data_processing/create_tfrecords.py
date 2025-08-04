# src/data_processing/create_tfrecords.py (最终平衡版)

import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    TRAINING_READY_DATA_PATH, TFRECORD_TRAIN_PATH, TFRECORD_VAL_PATH,
    SPLIT_INDICES_PATH
)

# --- TFRecord 转换辅助函数 (保持不变) ---
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(feature, label):
    feature_bytes = tf.io.serialize_tensor(feature)
    feature_proto = {
        'feature': _bytes_feature(feature_bytes),
        'label': _float_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_proto))
    return example_proto.SerializeToString()

def convert_to_tfrecord(h5_path, indices, output_path, dataset_type):
    """将HDF5中指定索引的数据集转换为TFRecord格式。"""
    print(f"开始转换 {len(indices)} 个 {dataset_type} 样本...")
    
    with h5py.File(h5_path, 'r') as hf:
        features_dset = hf[f'x_{dataset_type}']
        labels_dset = hf[f'y_{dataset_type}']
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in tqdm(range(len(indices)), desc=f"Writing {dataset_type} TFRecords"):
                # 使用传入的索引来获取正确的样本
                idx = indices[i]
                feature_sample = features_dset[idx, :, :]
                label_sample = labels_dset[idx]
                
                example = serialize_example(feature_sample, label_sample)
                writer.write(example)
    
    print(f"{dataset_type} 数据集已成功写入到: {output_path}")

def run_tfrecord_conversion():
    """执行HDF5到TFRecord的转换流程，并在训练集上进行平衡采样。"""
    if not os.path.exists(TRAINING_READY_DATA_PATH):
        print(f"错误: 未找到训练就绪文件 {TRAINING_READY_DATA_PATH}")
        print("请先运行 'normalize' 步骤。")
        return
        
    print("加载原始划分的索引和标签...")
    split_indices_data = np.load(SPLIT_INDICES_PATH)
    original_train_indices = split_indices_data['train_indices']
    original_val_indices = split_indices_data['val_indices']
    
    with h5py.File(TRAINING_READY_DATA_PATH, 'r') as hf:
        y_train_all = hf['y_train'][:]
        y_val_all = hf['y_val'][:]
    
    # --- 核心：对训练集进行平衡重采样 ---
    print("\n--- 开始对训练集进行平衡重采样 ---")
    df_train = pd.DataFrame({
        'original_index': np.arange(len(y_train_all)),
        'csi': y_train_all
    })
    
    # 1. 将CSI值分箱
    num_bins = 10  # 分成10个桶
    df_train['bin'] = pd.cut(df_train['csi'], bins=num_bins, labels=False, include_lowest=True)
    
    # 2. 确定每个箱子的样本量（以最少的箱子为准）
    bin_counts = df_train['bin'].value_counts()
    min_bin_size = bin_counts.min()
    print(f"CSI值被分为 {num_bins} 个箱子。")
    print(f"样本量最少的箱子有 {min_bin_size} 个样本。将以此为基准进行采样。")
    
    # 3. 从每个箱子中随机抽取 min_bin_size 个样本
    balanced_indices = []
    for bin_id in range(num_bins):
        # 确保箱子不为空
        if bin_counts.get(bin_id, 0) > 0:
            bin_samples = df_train[df_train['bin'] == bin_id]
            # 如果当前箱子的样本数少于min_bin_size，就全部取样（以防万一）
            sample_size = min(min_bin_size, len(bin_samples))
            balanced_indices.extend(
                bin_samples.sample(sample_size, random_state=42)['original_index'].tolist()
            )
            
    np.random.shuffle(balanced_indices) # 打乱最终顺序
    print(f"重采样后，新的平衡训练集大小为: {len(balanced_indices)} 个样本。\n")
    
    # --- 使用新的、平衡的索引来创建TFRecord文件 ---
    
    # 转换平衡后的训练集
    if not os.path.exists(TFRECORD_TRAIN_PATH):
        # 注意：这里的indices参数传入的是我们新生成的平衡索引
        convert_to_tfrecord(TRAINING_READY_DATA_PATH, balanced_indices, TFRECORD_TRAIN_PATH, 'train')
    else:
        print(f"TFRecord训练文件已存在于 {TFRECORD_TRAIN_PATH}, 跳过。")

    # 转换【完整】的验证集（验证集不需要采样）
    if not os.path.exists(TFRECORD_VAL_PATH):
        # 验证集使用原始的、完整的索引
        val_original_indices = np.arange(len(y_val_all))
        convert_to_tfrecord(TRAINING_READY_DATA_PATH, val_original_indices, TFRECORD_VAL_PATH, 'val')
    else:
        print(f"TFRecord验证文件已存在于 {TFRECORD_VAL_PATH}, 跳过。")