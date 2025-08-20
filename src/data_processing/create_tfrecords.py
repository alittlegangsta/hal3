# src/data_processing/create_tfrecords.py (最终可配置版)

import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 注意：我们不再从config.py直接导入路径，因为它们将通过函数参数传入

# --- TFRecord 转换辅助函数 (保持不变) ---
def _bytes_feature(value):
    """返回一个 bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """返回一个 float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(feature, label):
    """创建一个 tf.train.Example message a tf.train.Example message."""
    feature_bytes = tf.io.serialize_tensor(feature)
    feature_proto = {
        'feature': _bytes_feature(feature_bytes),
        'label': _float_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_proto))
    return example_proto.SerializeToString()

def convert_to_tfrecord(config, indices, output_path, dataset_type):
    """
    将HDF5中指定索引的数据集转换为TFRecord格式。
    现在从 config 字典获取输入文件路径。
    """
    h5_path = config['paths']['training_ready_data']
    print(f"开始转换 {len(indices)} 个 {dataset_type} 样本从 {os.path.basename(h5_path)}...")
    
    with h5py.File(h5_path, 'r') as hf:
        # 数据集名称在 HDF5 文件中是 'x_train', 'y_train', 'x_val', 'y_val'
        features_dset = hf[f'x_{dataset_type}']
        labels_dset = hf[f'y_{dataset_type}']
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in tqdm(range(len(indices)), desc=f"Writing {dataset_type} TFRecords for array_{config['array_id']}"):
                # 使用传入的索引来获取正确的样本
                # 这些索引是相对于 'x_train' 或 'x_val' 数据集内部的
                idx = indices[i]
                feature_sample = features_dset[idx, :, :]
                label_sample = labels_dset[idx]
                
                example = serialize_example(feature_sample, label_sample)
                writer.write(example)
    
    print(f"{dataset_type} 数据集已成功写入到: {output_path}")

def run_tfrecord_conversion(config):
    """
    执行HDF5到TFRecord的转换流程，并在训练集上进行平衡采样。
    
    Args:
        config (dict): 从 get_config() 函数生成的配置字典。
    """
    paths = config['paths']
    training_ready_path = paths['training_ready_data']
    tfrecord_train_path = paths['tfrecord_train']
    tfrecord_val_path = paths['tfrecord_val']
    split_indices_path = paths['split_indices'] # 注意：这个文件包含了全局索引

    if not os.path.exists(training_ready_path):
        print(f"错误: 未找到训练就绪文件 {training_ready_path}")
        print("请先运行 'normalize' 步骤。")
        return
        
    print(f"为阵列 '{config['array_id']}' 加载标签和索引...")
    
    with h5py.File(training_ready_path, 'r') as hf:
        # 这些是已经根据 split 步骤划分好的数据集
        y_train_all = hf['y_train'][:]
        y_val_all = hf['y_val'][:]
    
    # --- 核心：对训练集进行平衡重采样 ---
    print("\n--- 开始对训练集进行平衡重采样 ---")
    df_train = pd.DataFrame({
        'original_index': np.arange(len(y_train_all)), # 这是在 y_train 数组中的索引
        'csi': y_train_all
    })
    
    # 1. 将CSI值分箱
    num_bins = 10
    df_train['bin'] = pd.cut(df_train['csi'], bins=num_bins, labels=False, include_lowest=True)
    
    # 2. 确定每个箱子的样本量（以最少的箱子为准）
    bin_counts = df_train['bin'].value_counts()
    min_bin_size = bin_counts.min()
    print(f"CSI值被分为 {num_bins} 个箱子。")
    print(f"样本量最少的箱子有 {min_bin_size} 个样本。将以此为基准进行采样。")
    
    # 3. 从每个箱子中随机抽取 min_bin_size 个样本
    balanced_indices = []
    for bin_id in range(num_bins):
        if bin_counts.get(bin_id, 0) > 0:
            bin_samples = df_train[df_train['bin'] == bin_id]
            sample_size = min(min_bin_size, len(bin_samples))
            balanced_indices.extend(
                bin_samples.sample(sample_size, random_state=42)['original_index'].tolist()
            )
            
    np.random.shuffle(balanced_indices)
    print(f"重采样后，新的平衡训练集大小为: {len(balanced_indices)} 个样本。\n")
    
    # --- 使用新的、平衡的索引来创建TFRecord文件 ---
    
    # 转换平衡后的训练集
    if not os.path.exists(tfrecord_train_path):
        # 注意：这里的indices参数传入的是我们新生成的平衡索引
        convert_to_tfrecord(config, balanced_indices, tfrecord_train_path, 'train')
    else:
        print(f"TFRecord训练文件已存在于 {tfrecord_train_path}, 跳过。")

    # 转换【完整】的验证集（验证集不需要采样）
    if not os.path.exists(tfrecord_val_path):
        # 验证集使用原始的、完整的索引
        val_original_indices = np.arange(len(y_val_all))
        convert_to_tfrecord(config, val_original_indices, tfrecord_val_path, 'val')
    else:
        print(f"TFRecord验证文件已存在于 {tfrecord_val_path}, 跳过。")

# --- 用于直接运行此脚本进行调试 (可选) ---
if __name__ == '__main__':
    # 这是一个示例，展示如何独立运行此脚本
    # 正常情况下，此脚本应由 main.py 调用
    print("【调试模式】")
    # 动态导入 get_config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import get_config
    
    # 选择一个阵列进行测试
    test_array_id = '03' 
    print(f"为阵列 {test_array_id} 生成TFRecord文件...")
    
    # 获取该阵列的配置
    debug_config = get_config(test_array_id)
    
    # 运行转换
    run_tfrecord_conversion(debug_config)