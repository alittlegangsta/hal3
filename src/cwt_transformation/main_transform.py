# src/cwt_transformation/main_transform.py

import os
import sys
import numpy as np
import h5py
import pywt
from tqdm import tqdm

# 添加根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    ALIGNED_DATA_PATH, SCALOGRAM_H5_PATH, SAMPLING_RATE_HZ,
    CWT_WAVELET_NAME, CWT_SCALES_NUM, CWT_FREQ_RANGE_KHZ, CWT_BATCH_SIZE
)

def _calculate_scales():
    """计算用于CWT的尺度数组以匹配目标频率范围。"""
    f_min, f_max = CWT_FREQ_RANGE_KHZ[0] * 1000, CWT_FREQ_RANGE_KHZ[1] * 1000
    # 小波的中心频率
    central_freq = pywt.central_frequency(CWT_WAVELET_NAME, precision=8)
    # 计算对应频率范围的尺度
    scale_max = central_freq * SAMPLING_RATE_HZ / f_min
    scale_min = central_freq * SAMPLING_RATE_HZ / f_max
    # 生成对数间隔的尺度数组
    scales = np.geomspace(scale_min, scale_max, CWT_SCALES_NUM)
    return scales

def run_cwt_transformation():
    """
    执行阶段二：读取对齐的波形数据，进行CWT变换，并保存尺度图。
    """
    if not os.path.exists(ALIGNED_DATA_PATH):
        print(f"Error: Aligned data file not found at {ALIGNED_DATA_PATH}")
        print("Please run the 'preprocess' step first.")
        return

    if os.path.exists(SCALOGRAM_H5_PATH):
        print(f"CWT Result: Scalogram file already exists at {SCALOGRAM_H5_PATH}. Skipping.")
        return

    print("Loading aligned waveform data...")
    with h5py.File(ALIGNED_DATA_PATH, 'r') as hf:
        waveforms = hf['waveforms'][:]
    
    num_samples, num_timesteps = waveforms.shape
    scales = _calculate_scales()
    num_scales = len(scales)

    print(f"Total samples to process: {num_samples}")
    print(f"Wavelet: {CWT_WAVELET_NAME}, Scales: {num_scales}, Timesteps: {num_timesteps}")

    # 创建HDF5文件以分批写入结果
    with h5py.File(SCALOGRAM_H5_PATH, 'w') as hf:
        scalogram_dset = hf.create_dataset(
            'scalograms',
            shape=(num_samples, num_scales, num_timesteps),
            dtype=np.float32,
            chunks=(1, num_scales, num_timesteps) # 优化读取性能
        )

        # 分批处理以防止内存溢出
        for i in tqdm(range(0, num_samples, CWT_BATCH_SIZE), desc="Performing CWT"):
            batch_waveforms = waveforms[i:i + CWT_BATCH_SIZE]
            
            # pywt.cwt 返回 (coeffs, freqs)，我们只需要系数
            coeffs, _ = pywt.cwt(batch_waveforms, scales, CWT_WAVELET_NAME,
                                 sampling_period=1.0/SAMPLING_RATE_HZ)
            
            # 计算幅值并存入HDF5文件
            # coeffs的形状是 (num_scales, num_samples_in_batch, num_timesteps)
            # 需要调整轴的顺序为 (num_samples_in_batch, num_scales, num_timesteps)
            scalograms_batch = np.abs(np.moveaxis(coeffs, 0, 1))
            scalogram_dset[i:i + CWT_BATCH_SIZE, :, :] = scalograms_batch.astype(np.float32)

    print(f"Successfully saved scalograms to {SCALOGRAM_H5_PATH}")