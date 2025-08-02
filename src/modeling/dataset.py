import h5py
import numpy as np
import tensorflow as tf

def data_generator(h5_path, indices):
    """
    一个Python生成器，用于从HDF5文件中逐个产生数据样本。
    
    Args:
        h5_path (str): HDF5文件路径。
        indices (list or np.ndarray): 要产生的数据样本的索引列表。
    """
    with h5py.File(h5_path, 'r') as hf:
        scalograms = hf['scalograms']
        csi_labels = hf['csi_labels']
        for i in indices:
            # 产生 (特征, 标签) 对
            yield scalograms[i, :, :], csi_labels[i]

def create_dataset(h5_path, indices, config, is_training=True):
    """
    基于生成器创建一个 tf.data.Dataset 对象。
    
    Args:
        h5_path (str): HDF5文件路径。
        indices (list or np.ndarray): 数据集的索引。
        config (module): 配置模块。
        is_training (bool): 是否为训练集（决定是否打乱数据）。

    Returns:
        tf.data.Dataset: 配置好的数据集对象。
    """
    # 从HDF5文件中获取数据维度信息
    with h5py.File(h5_path, 'r') as hf:
        output_shape = hf['scalograms'].shape[1:]
        
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(h5_path, indices),
        output_signature=(
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    if is_training:
        # 对训练数据进行充分打乱
        dataset = dataset.shuffle(buffer_size=len(indices), reshuffle_each_iteration=True)
        
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset