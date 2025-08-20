# src/interpretation/plot_manual_analysis.py (最终可配置版)

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# --- 确保能找到 config ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import get_config # 导入新的配置函数

# ==============================================================================
# >>>>>>>>>> 您需要修改的区域：请在这里填入您的分析目标 <<<<<<<<<<<
# ==============================================================================
# 注意：这里的 'sample_index' 是相对于特定阵列实验的【验证集内部】的索引。
# 您可以通过运行 generate_analysis_candidates.py 来找到这些候选索引。
ANALYSIS_TARGETS = {
    "candidate_Excellent_index_4802": {
        "sample_index": 4802,
        "time_range_ms": [1.9, 2.1],
        "freq_range_khz": [23, 28]
    },
    "candidate_Good_index_8147": {
        "sample_index": 8147,
        "time_range_ms": [2, 2.2],
        "freq_range_khz": [5, 8]
    },
    "candidate_Poor_index_5400": {
        "sample_index": 5400,
        "time_range_ms": [1.7, 2.0],
        "freq_range_khz": [6, 10]
    },
    "candidate_Very Poor_index_8081": {
        "sample_index": 8081,
        "time_range_ms": [1.7, 2.1],
        "freq_range_khz": [5, 8]
    },
}
# ==============================================================================
# <<<<<<<<<<<<<<<<<<<<<<<<<< 修改区域结束 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================

def plot_manual_targets(config):
    """
    根据用户在 ANALYSIS_TARGETS 中定义的参数，进行滤波、高亮和绘图。
    
    Args:
        config (dict): 从 get_config() 函数生成的配置字典。
    """
    array_id = config['array_id']
    print(f"--- [开始为阵列 {array_id} 进行手动指定区域的分析与绘图] ---")
    
    # 1. 从配置加载资源
    paths = config['paths']
    physical_params = config['physical']
    sampling_rate_hz = physical_params['sampling_rate_hz']
    
    # --- 核心修改：使用动态路径 ---
    output_dir = os.path.join(paths['plot_dir'], 'manual_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果图将被保存在: {output_dir}")
    
    print("正在加载索引映射和数据文件...")
    if not os.path.exists(paths['split_indices']) or not os.path.exists(paths['aligned_data']):
        print(f"错误: 缺少必要文件。请先为阵列 {array_id} 运行 'preprocess' 和 'split' 步骤。")
        return

    split_indices = np.load(paths['split_indices'])
    val_indices = split_indices['val_indices'] # 这是从“验证集内部索引”到“HDF5全局索引”的映射

    # 2. 遍历所有目标并作图
    # 我们将HDF5文件的打开放在循环外部，以提高效率
    with h5py.File(paths['aligned_data'], 'r') as hf:
        waveforms_dset = hf['waveforms']
        
        for name, params in ANALYSIS_TARGETS.items():
            internal_val_index = params["sample_index"]
            
            # 检查内部索引是否有效
            if not (0 <= internal_val_index < len(val_indices)):
                print(f"错误：样本序号 {internal_val_index} 超出验证集范围 (0-{len(val_indices)-1})，已跳过。")
                continue
            
            # 使用内部索引找到全局索引，然后精确加载单个波形
            global_hdf5_index = val_indices[internal_val_index]
            waveform = waveforms_dset[global_hdf5_index]
            
            t_min, t_max = params["time_range_ms"]
            f_min, f_max = params["freq_range_khz"]
            
            # 3. 进行带通滤波
            low_cut_hz = f_min * 1000
            high_cut_hz = f_max * 1000
            sos = butter(4, [low_cut_hz, high_cut_hz], btype='band', fs=sampling_rate_hz, output='sos')
            filtered_waveform = sosfiltfilt(sos, waveform)

            # 4. 绘图
            fig, ax = plt.subplots(figsize=(15, 6))
            time_ms = np.arange(len(waveform)) * (1 / sampling_rate_hz) * 1000
            
            ax.plot(time_ms, filtered_waveform, 'r', label='Filtered Waveform')
            ax.axvspan(t_min, t_max, color='yellow', alpha=0.4, label=f'Highlighted Time Zone')
            
            # --- 核心修改：在标题中加入阵列ID ---
            title = (f"Manually Focused Analysis for Array {array_id} - Val Sample Index: {internal_val_index}\n"
                     f"(Global HDF5 Index: {global_hdf5_index})\n"
                     f"Time Window: {t_min:.1f}-{t_max:.1f} ms | "
                     f"Frequency Band: {f_min:.1f}-{f_max:.1f} kHz")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Time (ms)", fontsize=14)
            ax.set_ylabel("Amplitude", fontsize=14)
            ax.set_xlim(0, 4)
            ax.grid(True)
            ax.legend()
            
            output_path = os.path.join(output_dir, f"{name}.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  - 已生成图表: {output_path}")

    print(f"\n--- [阵列 {array_id} 的手动分析绘图完毕] ---")

if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="对指定接收器阵列的特定样本进行手动聚焦分析和绘图。")
    parser.add_argument(
        '--array',
        type=str,
        default='03',
        help="指定要分析的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    args = parser.parse_args()

    # 根据命令行参数获取配置
    config = get_config(args.array)
    
    # 运行主函数
    plot_manual_targets(config)