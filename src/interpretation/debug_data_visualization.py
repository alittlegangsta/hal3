# src/interpretation/debug_data_visualization.py

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt  # 导入PyWavelets库用于频率计算

# --- 确保能找到 config 和其他模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import (
    load_config, 
    SAMPLING_RATE_HZ, 
    CWT_WAVELET_NAME, 
    CWT_SCALES_NUM, 
    CWT_FREQ_RANGE_KHZ
)

def _calculate_cwt_frequencies_khz():
    """
    根据config中的CWT参数，计算每个尺度对应的物理频率值 (单位: kHz)。
    """
    f_min_hz, f_max_hz = CWT_FREQ_RANGE_KHZ[0] * 1000, CWT_FREQ_RANGE_KHZ[1] * 1000
    central_freq = pywt.central_frequency(CWT_WAVELET_NAME, precision=8)
    
    # 根据频率反算尺度
    scale_max = central_freq * SAMPLING_RATE_HZ / f_min_hz
    scale_min = central_freq * SAMPLING_RATE_HZ / f_max_hz
    
    scales = np.geomspace(scale_min, scale_max, CWT_SCALES_NUM)
    
    # 根据尺度计算频率
    frequencies_hz = pywt.scale2frequency(CWT_WAVELET_NAME, scales) * SAMPLING_RATE_HZ
    return frequencies_hz / 1000 # 转换为 kHz

def visualize_csi_extremes():
    """
    可视化CSI值处于两个极端的样本，使用英文标签和物理单位坐标轴。
    """
    print("--- [Starting Data Visualization Diagnosis] ---")
    
    # 1. 加载配置和最终的训练就绪数据
    config = load_config()
    paths = config['paths']
    
    if not os.path.exists(paths['training_ready_data']):
        print(f"Error: Training-ready data file not found: {paths['training_ready_data']}")
        print("Please ensure the 'normalize' step has completed successfully.")
        return
        
    print("Loading normalized validation data...")
    with h5py.File(paths['training_ready_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    print("Data loaded.")
    
    # 2. 寻找极端样本的索引
    good_bond_indices = np.where(y_val < 0.05)[0]
    poor_bond_indices = np.where(y_val > 0.95)[0]
    
    if len(good_bond_indices) < 3 or len(poor_bond_indices) < 3:
        print("Warning: Fewer than 3 examples found for good or poor bond cases. Comparison may be limited.")
        return

    np.random.shuffle(good_bond_indices)
    np.random.shuffle(poor_bond_indices)
    
    sample_indices_good = good_bond_indices[:3]
    sample_indices_poor = poor_bond_indices[:3]

    # 3. 计算坐标轴
    print("Calculating physical axes (time and frequency)...")
    num_timesteps = x_val.shape[2]
    sampling_period_ms = (1 / SAMPLING_RATE_HZ) * 1000
    time_axis_ms = np.arange(num_timesteps) * sampling_period_ms
    freq_axis_khz = _calculate_cwt_frequencies_khz()

    # 4. 绘制对比图
    print("Plotting comparison figure...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 10), sharex=True, sharey=True)
    fig.suptitle("CWT Scalogram Comparison: Good Bond vs. Poor Bond", fontsize=22, y=0.98)

    # 定义图像范围以正确显示坐标轴
    extent = [time_axis_ms.min(), time_axis_ms.max(), freq_axis_khz.min(), freq_axis_khz.max()]

    # 绘制良好胶结的样本
    for i, idx in enumerate(sample_indices_good):
        ax = axes[0, i]
        im = ax.imshow(x_val[idx], aspect='auto', cmap='jet', origin='lower', extent=extent)
        ax.set_title(f"Good Bond Example\nTrue CSI = {y_val[idx]:.3f}", fontsize=16)
        if i == 0:
            ax.set_ylabel("Frequency (kHz)", fontsize=14)

    # 绘制严重窜槽的样本
    for i, idx in enumerate(sample_indices_poor):
        ax = axes[1, i]
        im = ax.imshow(x_val[idx], aspect='auto', cmap='jet', origin='lower', extent=extent)
        ax.set_title(f"Poor Bond Example\nTrue CSI = {y_val[idx]:.3f}", fontsize=16)
        ax.set_xlabel("Time (ms)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Frequency (kHz)", fontsize=14)
            
    # 添加一个共享的颜色条
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
            
    output_path = os.path.join(paths['plot_dir'], 'debug_csi_extremes_comparison_en.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic figure has been saved to: {output_path}")
    print("Please open this figure to analyze the results.")

if __name__ == '__main__':
    visualize_csi_extremes()