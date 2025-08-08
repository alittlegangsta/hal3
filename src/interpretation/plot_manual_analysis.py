# src/interpretation/plot_manual_analysis.py (最终修正版)

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# --- 确保能找到 config ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from config import load_config, SAMPLING_RATE_HZ

# ==============================================================================
# >>>>>>>>>> 您需要修改的区域：请在这里填入您的分析目标 <<<<<<<<<<<
# ==============================================================================
ANALYSIS_TARGETS = {
    "candidate_Excellent_index_4802": {
        "sample_index": 4802,
        "time_range_ms": [1.9, 2.1],
        "freq_range_khz": [23, 28]
    },
    "candidate_Excellent_index_9495": {
        "sample_index": 9495,
        "time_range_ms": [1.4, 1.6],
        "freq_range_khz": [8, 12]
    },
    "candidate_Excellent_index_15183": {
        "sample_index": 15183,
        "time_range_ms": [1.6, 1.8],
        "freq_range_khz": [8, 12]
    },
    "candidate_Good_index_537": {
        "sample_index": 537,
        "time_range_ms": [1.2, 1.5],
        "freq_range_khz": [7, 11]
    },
    "candidate_Good_index_8147": {
        "sample_index": 8147,
        "time_range_ms": [2, 2.2],
        "freq_range_khz": [5, 8]
    },
    "candidate_Good_index_10703": {
        "sample_index": 10703,
        "time_range_ms": [0.5, 0.7],
        "freq_range_khz": [8, 10]
    },
    "candidate_Poor_index_2576": {
        "sample_index": 2576,
        "time_range_ms": [0.9, 1.1],
        "freq_range_khz": [15, 18]
    },
    "candidate_Poor_index_4123": {
        "sample_index": 4123,
        "time_range_ms": [1.3, 1.6],
        "freq_range_khz": [5, 9]
    },
    "candidate_Poor_index_5400": {
        "sample_index": 5400,
        "time_range_ms": [1.7, 2.0],
        "freq_range_khz": [6, 10]
    },
    "candidate_Very Poor_index_2352": {
        "sample_index": 2352,
        "time_range_ms": [0.1, 0.3],
        "freq_range_khz": [16, 20]
    },
    "candidate_Very Poor_index_4624": {
        "sample_index": 4624,
        "time_range_ms": [0.4, 0.6],
        "freq_range_khz": [16, 20]
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

def plot_manual_targets():
    """
    根据用户在 ANALYSIS_TARGETS 中定义的参数，进行滤波、高亮和绘图。
    """
    print("--- [开始进行手动指定区域的分析与绘图] ---")
    
    # 1. 加载资源
    config = load_config()
    paths = config['paths']
    output_dir = os.path.join(paths['plot_dir'], 'manual_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"分析结果图将被保存在: {output_dir}")
    
    print("正在加载索引映射和数据文件...")
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
            
            # --- 核心修正：使用内部索引找到全局索引，然后精确加载单个波形 ---
            global_hdf5_index = val_indices[internal_val_index]
            waveform = waveforms_dset[global_hdf5_index]
            # -----------------------------------------------------------------
            
            t_min, t_max = params["time_range_ms"]
            f_min, f_max = params["freq_range_khz"]
            
            # 3. 进行带通滤波
            low_cut_hz = f_min * 1000
            high_cut_hz = f_max * 1000
            sos = butter(4, [low_cut_hz, high_cut_hz], btype='band', fs=SAMPLING_RATE_HZ, output='sos')
            filtered_waveform = sosfiltfilt(sos, waveform)

            # 4. 绘图
            fig, ax = plt.subplots(figsize=(15, 6))
            time_ms = np.arange(len(waveform)) * (1 / SAMPLING_RATE_HZ) * 1000
            
            ax.plot(time_ms, filtered_waveform, 'r', label='Filtered Waveform')
            ax.axvspan(t_min, t_max, color='yellow', alpha=0.4, label=f'Highlighted Time Zone')
            
            title = (f"Manually Focused Analysis for Validation Sample Index: {internal_val_index}\n"
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

    print("\n--- [手动分析绘图完毕] ---")

if __name__ == '__main__':
    plot_manual_targets()