# src/modeling/dataset.py

import numpy as np
import h5py

class DataGenerator:
    """
    为Keras模型生成数据的生成器。
    在生成每个批次时动态地进行标准化。
    """
    def __init__(self, h5_path, indices, csi_labels, batch_size, norm_stats, shuffle=True):
        self.h5_path = h5_path
        self.indices = indices
        self.csi_labels = csi_labels
        self.batch_size = batch_size
        self.norm_stats = norm_stats
        self.shuffle = shuffle
        self.num_samples = len(self.indices)
        self.mean = self.norm_stats['mean']
        self.std = self.norm_stats['std']
        
    def __len__(self):
        """返回每个epoch的批次数。"""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __call__(self):
        """TensorFlow Dataset.from_generator 需要一个可调用对象。"""
        # 在每个epoch开始时，如果需要，打乱索引
        epoch_indices = np.copy(self.indices)
        if self.shuffle:
            np.random.shuffle(epoch_indices)
        
        with h5py.File(self.h5_path, 'r') as hf:
            scalograms_dset = hf['scalograms']
            
            for i in range(0, self.num_samples, self.batch_size):
                batch_indices = epoch_indices[i:i + self.batch_size]
                
                # h5py需要排序后的索引以获得最佳性能
                sorted_batch_indices = np.sort(batch_indices)
                
                # 读取一批数据
                batch_scalograms = scalograms_dset[sorted_batch_indices, :, :]
                batch_csi = self.csi_labels[sorted_batch_indices]
                
                # --- 核心：执行标准化 ---
                normalized_scalograms = (batch_scalograms - self.mean) / self.std
                
                # 为模型输入增加一个通道维度 (..., 1)
                yield np.expand_dims(normalized_scalograms, axis=-1), batch_csi