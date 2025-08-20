# src/interpretation/debug_data_visualization.py (最终可配置版)

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt

# --- 确保能找到 config 和其他模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import get_config

def _calculate_cwt_frequencies_khz(config):
    """
    根据config中的CWT参数，计算每个尺度对应的物理频率值 (单位: kHz)。
    """
    params = config['physical']
    cwt_freq_range_khz = params['cwt_freq_range_khz']
    cwt_wavelet_name = params['cwt_wavelet_name']
    sampling_rate_hz = params['sampling_rate_hz']
    cwt_scales_num = params['cwt_scales_num']

    f_min_hz, f_max_hz = cwt_freq_range_khz[0] * 1000, cwt_freq_range_khz[1] * 1000
    central_freq = pywt.central_frequency(cwt_wavelet_name, precision=8)
    
    scale_max = central_freq * sampling_rate_hz / f_min_hz
    scale_min = central_freq * sampling_rate_hz / f_max_hz
    
    scales = np.geomspace(scale_min, scale_max, cwt_scales_num)
    
    frequencies_hz = pywt.scale2frequency(cwt_wavelet_name, scales) * sampling_rate_hz
    return frequencies_hz / 1000

def visualize_csi_extremes(config):
    """
    可视化CSI值处于两个极端的样本，使用英文标签和物理单位坐标轴。
    
    Args:
        config (dict): 从 get_config() 函数生成的配置字典。
    """
    array_id = config['array_id']
    print(f"--- [为阵列 {array_id} 开始数据可视化诊断] ---")
    
    # 1. 从配置加载资源
    paths = config['paths']
    physical_params = config['physical']
    
    if not os.path.exists(paths['training_ready_data']):
        print(f"错误: 训练就绪数据文件未找到: {paths['training_ready_data']}")
        print(f"请确保已为阵列 {array_id} 成功运行了 'normalize' 步骤。")
        return
        
    print("加载标准化的验证集数据...")
    with h5py.File(paths['training_ready_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    print("数据加载完毕。")
    
    # 2. 寻找极端样本的索引
    good_bond_indices = np.where(y_val < 0.05)[0]
    poor_bond_indices = np.where(y_val > 0.95)[0]
    
    if len(good_bond_indices) < 3 or len(poor_bond_indices) < 3:
        print("警告: 良好或差胶结的极端样本少于3个，对比可能受限。")
        # 即使样本少，也继续运行
        if len(good_bond_indices) == 0 or len(poor_bond_indices) == 0:
            print("错误：至少一类极端样本数量为0，无法生成对比图。")
            return

    np.random.shuffle(good_bond_indices)
    np.random.shuffle(poor_bond_indices)
    
    sample_indices_good = good_bond_indices[:3]
    sample_indices_poor = poor_bond_indices[:3]

    # 3. 计算坐标轴
    print("计算物理坐标轴 (时间和频率)...")
    num_timesteps = x_val.shape[2]
    sampling_period_ms = (1 / physical_params['sampling_rate_hz']) * 1000
    time_axis_ms = np.arange(num_timesteps) * sampling_period_ms
    freq_axis_khz = _calculate_cwt_frequencies_khz(config)

    # 4. 绘制对比图
    print("正在绘制对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 10), sharex=True, sharey=True)
    fig.suptitle(f"CWT Scalogram Comparison for Array {array_id}: Good vs. Poor Bond", fontsize=22, y=0.98)

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
            
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
            
    output_path = os.path.join(paths['plot_dir'], f'debug_csi_extremes_comparison_array_{array_id}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n诊断图已保存至: {output_path}")
    print("请打开此图以分析结果。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为指定接收器阵列可视化CSI值两端的样本。")
    parser.add_argument(
        '--array',
        type=str,
        default='03',
        help="指定要分析的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    args = parser.parse_args()

    config = get_config(args.array)
    visualize_csi_extremes(config)