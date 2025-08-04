# src/interpretation/run_analysis.py (最终修正版)

import os
import sys
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
import pywt # 导入 PyWavelets 库

# --- 确保能找到 config 和其他模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import (
    load_config, SAMPLING_RATE_HZ, CWT_WAVELET_NAME, 
    CWT_SCALES_NUM, CWT_FREQ_RANGE_KHZ, DATA_PROCESSING_CONFIG
)
# 假设grad_cam.py与此文件在同一目录下
from src.interpretation.grad_cam import make_gradcam_heatmap


# --- 关键修正：将缺失的函数直接定义在这里 ---
def _calculate_cwt_frequencies_khz():
    """
    根据config中的CWT参数，计算每个尺度对应的物理频率值 (单位: kHz)。
    """
    f_min_hz, f_max_hz = CWT_FREQ_RANGE_KHZ[0] * 1000, CWT_FREQ_RANGE_KHZ[1] * 1000
    central_freq = pywt.central_frequency(CWT_WAVELET_NAME, precision=8)
    scale_max = central_freq * SAMPLING_RATE_HZ / f_min_hz
    scale_min = central_freq * SAMPLING_RATE_HZ / f_max_hz
    scales = np.geomspace(scale_min, scale_max, CWT_SCALES_NUM)
    frequencies_hz = pywt.scale2frequency(CWT_WAVELET_NAME, scales) * SAMPLING_RATE_HZ
    return frequencies_hz / 1000 # 转换为 kHz
# -----------------------------------------------


def run_analysis():
    """
    执行一套完整的、用于深度分析模型性能和可解释性的可视化流程。
    """
    print("--- [开始最终模型分析] ---")
    
    # 1. 加载配置、模型和数据
    print("[步骤 1/5] 正在加载配置、模型和数据...")
    config = load_config()
    paths = config['paths']
    model_params = config['modeling']
    
    model = load_model(paths['model_checkpoint'], compile=False)
    
    with h5py.File(paths['training_ready_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    
    # 加载未经CWT变换的原始对齐波形数据
    with h5py.File(paths['aligned_data'], 'r') as hf:
        split_indices = np.load(paths['split_indices'])
        val_indices = split_indices['val_indices']
        # 确保索引是有序的，以匹配 y_val 的顺序
        sorted_val_indices = np.sort(val_indices)
        original_waveforms_val = hf['waveforms'][sorted_val_indices]

    x_val_processed = np.expand_dims(x_val, axis=-1)
    print("所有数据加载完毕。")

    # 2. 生成预测
    print("[步骤 2/5] 正在验证集上生成预测...")
    y_pred = model.predict(x_val_processed, batch_size=model_params['batch_size']).flatten()
    print("预测已生成。")

    # 3. 创建分析用的DataFrame
    df_val = pd.DataFrame({'true_csi': y_val, 'pred_csi': y_pred})
    
    bins = [c[1] for c in DATA_PROCESSING_CONFIG['csi_bins']] + [1.0]
    labels = [c[0] for c in DATA_PROCESSING_CONFIG['csi_bins']]
    df_val['quality'] = pd.cut(df_val['true_csi'], bins=bins, labels=labels, include_lowest=True, right=True)

    # 4. 生成所有分析图表
    print("[步骤 3/5] 正在生成性能分析图...")
    output_summary_path = os.path.join(paths['plot_dir'], 'performance_summary_plots.png')
    plot_performance_summary(df_val, labels, output_summary_path)
    print(f"  - 性能汇总图已保存至: {output_summary_path}")

    print("[步骤 4/5] 正在生成平均注意力和方差图...")
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    mean_map, var_map = calculate_mean_attention_maps(model, x_val_processed, last_conv_layer_name)
    attention_maps_path = os.path.join(paths['plot_dir'], 'attention_mean_variance.png')
    plot_attention_maps(mean_map, var_map, attention_maps_path)
    print(f"  - 平均注意力图已保存至: {attention_maps_path}")

    print("[步骤 5/5] 正在生成多维度样本对比图...")
    multi_dim_plot_path = os.path.join(paths['plot_dir'], 'multi_dimension_sample_analysis.png')
    plot_multidim_samples(df_val, x_val, original_waveforms_val, model, last_conv_layer_name, multi_dim_plot_path)
    print(f"  - 多维度样本对比图已保存至: {multi_dim_plot_path}")

    print("\n--- [分析成功完成] ---")

# ... (后续的所有绘图函数 plot_performance_summary, calculate_mean_attention_maps 等保持不变) ...
def plot_performance_summary(df, quality_labels, save_path):
    """绘制包含散点图、样本分布、误差分布的性能汇总图。"""
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, :]) # 散点图
    ax2 = fig.add_subplot(gs[1, 0]) # 样本分布
    ax3 = fig.add_subplot(gs[1, 1]) # 误差分布
    ax4 = fig.add_subplot(gs[2, :]) # 按类别误差分布

    colors = plt.cm.get_cmap('viridis', len(quality_labels))
    
    # 1. 散点图
    for i, quality in enumerate(quality_labels):
        subset = df[df['quality'] == quality]
        ax1.scatter(subset['true_csi'], subset['pred_csi'], color=colors(i), label=quality, alpha=0.6, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('True CSI', fontsize=14)
    ax1.set_ylabel('Predicted CSI', fontsize=14)
    ax1.set_title('Predicted vs. True CSI by Cement Quality', fontsize=16)
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')

    # 2. 样本分布
    quality_counts = df['quality'].value_counts().loc[quality_labels]
    quality_counts.plot(kind='bar', ax=ax2, color=[colors(i) for i in range(len(quality_labels))])
    ax2.set_title('Sample Distribution by Cement Quality', fontsize=16)
    ax2.set_ylabel('Number of Samples', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)

    # 3. 总体误差分布
    df['error'] = df['pred_csi'] - df['true_csi']
    df['error'].plot(kind='hist', bins=50, ax=ax3, color='gray')
    ax3.set_title('Overall Prediction Error (Pred - True)', fontsize=16)
    ax3.set_xlabel('Prediction Error', fontsize=14)
    
    # 4. 按类别误差分布
    for i, quality in enumerate(quality_labels):
        df[df['quality'] == quality]['error'].plot(kind='hist', bins=30, ax=ax4, alpha=0.7, label=quality, color=colors(i))
    ax4.set_title('Prediction Error Distribution by Cement Quality', fontsize=16)
    ax4.set_xlabel('Prediction Error', fontsize=14)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def calculate_mean_attention_maps(model, data, layer_name):
    """计算平均Grad-CAM注意力和方差。"""
    all_heatmaps = []
    subset_indices = np.random.choice(len(data), min(1000, len(data)), replace=False)
    for i in tqdm(subset_indices, desc="Calculating Grad-CAM for Attention Maps"):
        img_array = data[i:i+1]
        heatmap = make_gradcam_heatmap(img_array, model, layer_name)
        all_heatmaps.append(heatmap)
    
    all_heatmaps = np.array(all_heatmaps)
    mean_map = np.mean(all_heatmaps, axis=0)
    var_map = np.var(all_heatmaps, axis=0)
    return mean_map, var_map

def plot_attention_maps(mean_map, var_map, save_path):
    """绘制平均注意力和方差图。"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    freqs_khz = _calculate_cwt_frequencies_khz()
    time_ms = np.arange(mean_map.shape[1]) * (1 / SAMPLING_RATE_HZ) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

    im = axes[0].imshow(mean_map, cmap='jet', aspect='auto', origin='lower', extent=extent)
    axes[0].set_title('Mean Attention Map (Across All Validation Samples)', fontsize=16)
    axes[0].set_ylabel('Frequency (kHz)')
    axes[0].set_xlabel('Time (ms)')
    fig.colorbar(im, ax=axes[0])
    
    im = axes[1].imshow(var_map, cmap='magma', aspect='auto', origin='lower', extent=extent)
    axes[1].set_title('Attention Variance Map', fontsize=16)
    axes[1].set_xlabel('Time (ms)')
    fig.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_multidim_samples(df, x_val, waveforms, model, layer_name, save_path):
    """绘制多维度样本对比图。"""
    quality_labels = df['quality'].cat.categories
    fig, axes = plt.subplots(len(quality_labels), 3, figsize=(24, 8 * len(quality_labels)))
    fig.suptitle('Multi-dimensional Sample Analysis', fontsize=24, y=1.02)

    freqs_khz = _calculate_cwt_frequencies_khz()
    time_ms = np.arange(waveforms.shape[1]) * (1 / SAMPLING_RATE_HZ) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

    for i, quality in enumerate(quality_labels):
        subset_indices = df[df['quality'] == quality].index
        if len(subset_indices) == 0:
            print(f"Warning: No samples found for quality '{quality}'. Skipping this row in the plot.")
            for j in range(3): axes[i, j].axis('off')
            continue
        sample_idx = np.random.choice(subset_indices)
        
        # 1. Grad-CAM图
        img_array = np.expand_dims(x_val[sample_idx], axis=(0, -1))
        heatmap = make_gradcam_heatmap(img_array, model, layer_name)
        axes[i, 0].imshow(x_val[sample_idx], cmap='jet', extent=extent, aspect='auto', origin='lower')
        axes[i, 0].imshow(heatmap, cmap='hot', alpha=0.5, extent=extent, aspect='auto', origin='lower')
        title = f'Quality: {quality}\nTrue CSI: {df.loc[sample_idx, "true_csi"]:.2f}, Pred CSI: {df.loc[sample_idx, "pred_csi"]:.2f}'
        axes[i, 0].set_title(f"Grad-CAM Heatmap\n{title}", fontsize=14)
        axes[i, 0].set_ylabel("Frequency (kHz)", fontsize=12)

        # 2. 原始波形
        axes[i, 1].plot(time_ms, waveforms[sample_idx])
        axes[i, 1].set_title(f"Original Waveform\n{title}", fontsize=14)
        axes[i, 1].grid(True)
        
        # 3. 滤波后的波形
        max_attention_freq_idx = np.argmax(np.mean(heatmap, axis=1))
        center_freq = freqs_khz[max_attention_freq_idx]
        band_width = 5 
        low_cut = max(0.1, center_freq - band_width/2) * 1000
        high_cut = (center_freq + band_width/2) * 1000
        
        sos = butter(4, [low_cut, high_cut], btype='band', fs=SAMPLING_RATE_HZ, output='sos')
        filtered_waveform = sosfiltfilt(sos, waveforms[sample_idx])
        axes[i, 2].plot(time_ms, filtered_waveform, 'r')
        axes[i, 2].set_title(f"Filtered to Attention Band ({low_cut/1000:.1f}-{high_cut/1000:.1f} kHz)\n{title}", fontsize=14)
        axes[i, 2].grid(True)
        
    for ax in axes[-1, :]: # Only set x-label for the bottom row
        ax.set_xlabel("Time (ms)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    run_analysis()