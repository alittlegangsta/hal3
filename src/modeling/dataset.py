# src/modeling/dataset.py (更新版)

import h5py
import numpy as np
import tensorflow as tf

def data_generator(h5_path, indices, mean, std):
    """
    一个Python生成器，用于产生经过标准化处理的数据样本。
    """
    with h5py.File(h5_path, 'r') as hf:
        scalograms = hf['scalograms']
        csi_labels = hf['csi_labels']
        for i in indices:
            scalogram = scalograms[i, :, :]
            # --- 新增：应用log1p和标准化 ---
            scalogram_log = np.log1p(scalogram)
            scalogram_norm = (scalogram_log - mean) / (std + 1e-8) # 加一个epsilon防止除以零
            
            yield scalogram_norm, csi_labels[i]

def create_dataset(h5_path, indices, batch_size, mean, std, is_training=True):
    """
    基于生成器创建一个 tf.data.Dataset 对象。
    """
    with h5py.File(h5_path, 'r') as hf:
        output_shape = hf['scalograms'].shape[1:]
        
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(h5_path, indices, mean, std), # 传递mean和std
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset