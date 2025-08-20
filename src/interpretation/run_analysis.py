# src/interpretation/run_analysis.py (最终高级分析版)

import os
import sys
import argparse
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pywt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import get_config
from src.interpretation.grad_cam import make_gradcam_heatmap, superimpose_gradcam

def _calculate_cwt_frequencies_khz(config):
    """根据配置计算CWT频率轴。"""
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

def run_analysis(config):
    """
    执行最终的模型分析、评估和可视化，包含按类别的注意力分析。
    """
    array_id = config['array_id']
    print(f"--- [开始为阵列 {array_id} 进行最终模型分析] ---")
    
    paths = config['paths']
    model_params = config['modeling']
    data_proc_params = config['data_processing']

    print("加载模型和数据...")
    if not all(os.path.exists(p) for p in [paths['model_checkpoint'], paths['training_ready_data'], paths['aligned_data'], paths['split_indices']]):
        print(f"错误: 缺少一个或多个必要文件。请确保已为阵列 {array_id} 完整运行了所有前期步骤。")
        return
        
    model = load_model(paths['model_checkpoint'], compile=False)
    with h5py.File(paths['training_ready_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    with h5py.File(paths['aligned_data'], 'r') as hf:
        split_indices = np.load(paths['split_indices'])
        val_indices = split_indices['val_indices']
        sorted_val_indices = np.sort(val_indices)
        original_waveforms_val = hf['waveforms'][sorted_val_indices]

    x_val_processed = np.expand_dims(x_val, axis=-1)
    print("所有数据加载完毕。")

    print("在验证集上进行预测...")
    y_pred = model.predict(x_val_processed, batch_size=model_params['batch_size']).flatten()
    df_val = pd.DataFrame({'true_csi': y_val, 'pred_csi': y_pred})
    
    bins = [c[1] for c in data_proc_params['csi_bins']] + [1.0]
    labels = [c[0] for c in data_proc_params['csi_bins']]
    df_val['quality'] = pd.cut(df_val['true_csi'], bins=bins, labels=labels, include_lowest=True, right=True)

    # --- 绘图与分析流程 ---
    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    
    # 1. 绘制整体性能图
    output_summary_path = os.path.join(paths['plot_dir'], 'performance_summary_plots.png')
    plot_performance_summary(df_val, labels, output_summary_path, config)
    print(f"  - 性能汇总图已保存至: {output_summary_path}")
    
    # 2. 绘制多维度样本对比图
    # 将 original_index 添加到 df_val 以便在绘图函数中溯源
    df_val['original_index'] = np.arange(len(y_val))
    multi_dim_plot_path = os.path.join(paths['plot_dir'], 'multi_dimension_sample_analysis.png')
    plot_multidim_samples(df_val, x_val, original_waveforms_val, model, last_conv_layer_name, multi_dim_plot_path, config)
    print(f"  - 多维度样本对比图已保存至: {multi_dim_plot_path}")

    # 3. *** 新增：执行按类别的深度注意力分析 ***
    class_attention_path = os.path.join(paths['plot_dir'], 'class_conditional_attention_analysis.png')
    analyze_attention_by_quality(df_val, x_val_processed, model, last_conv_layer_name, class_attention_path, config)
    print(f"  - 按类别的深度注意力分析图已保存至: {class_attention_path}")

    print(f"\n--- [阵列 {array_id} 的分析成功完成] ---")


# ==============================================================================
# 新增：按类别进行注意力分析的核心函数
# ==============================================================================

def analyze_attention_by_quality(df_val, x_val_processed, model, layer_name, save_path, config):
    """
    计算并可视化每个胶结质量类别的平均注意力、方差以及差异图。
    """
    quality_labels = df_val['quality'].cat.categories.tolist()
    attention_stats = {}

    for quality in quality_labels:
        print(f"\n正在为 '{quality}' 类别计算注意力图...")
        indices = df_val[df_val['quality'] == quality].index.tolist()
        
        if len(indices) == 0:
            print(f"  - 警告: '{quality}' 类别没有样本，跳过。")
            continue
        
        # 为了效率，每个类别最多使用500个样本进行计算
        if len(indices) > 500:
            indices = np.random.choice(indices, 500, replace=False)
            
        heatmaps = []
        for i in tqdm(indices, desc=f"  - Processing {quality}"):
            img_array = x_val_processed[i:i+1]
            heatmap = make_gradcam_heatmap(img_array, model, layer_name)
            heatmaps.append(heatmap.astype(np.float32))
        
        if heatmaps:
            heatmaps = np.array(heatmaps)
            attention_stats[quality] = {
                'mean': np.mean(heatmaps, axis=0),
                'var': np.var(heatmaps, axis=0),
                'count': len(indices)
            }

    plot_attention_by_quality(attention_stats, save_path, config)

def plot_attention_by_quality(stats, save_path, config):
    """
    为按类别分析的结果生成一个综合图表。
    """
    labels = list(stats.keys())
    num_labels = len(labels)
    if num_labels == 0: return

    # 动态设置图表尺寸
    fig = plt.figure(figsize=(24, 6 * num_labels + 12))
    gs = fig.add_gridspec(num_labels + 2, 2) # 为差异图增加两行
    fig.suptitle(f"Class-Conditional Attention Analysis for Array {config['array_id']}", fontsize=24, y=0.99)
    
    phys_params = config['physical']
    freqs_khz = _calculate_cwt_frequencies_khz(config)
    time_ms = np.arange(phys_params['waveform_timesteps']) * (1 / phys_params['sampling_rate_hz']) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]
    
    # 绘制每个类别的平均和方差图
    for i, label in enumerate(labels):
        ax_mean = fig.add_subplot(gs[i, 0])
        ax_var = fig.add_subplot(gs[i, 1])
        
        im_mean = ax_mean.imshow(stats[label]['mean'], cmap='jet', aspect='auto', origin='lower', extent=extent)
        ax_mean.set_title(f"Mean Attention: '{label}' (n={stats[label]['count']})", fontsize=16)
        ax_mean.set_ylabel("Frequency (kHz)")
        fig.colorbar(im_mean, ax=ax_mean)

        im_var = ax_var.imshow(stats[label]['var'], cmap='plasma', aspect='auto', origin='lower', extent=extent)
        ax_var.set_title(f"Attention Variance: '{label}'", fontsize=16)
        fig.colorbar(im_var, ax=ax_var)

    # 绘制差异注意力图
    if 'Very Poor' in stats and 'Excellent' in stats:
        ax_diff1 = fig.add_subplot(gs[num_labels, :])
        diff_map1 = stats['Very Poor']['mean'] - stats['Excellent']['mean']
        vmax = np.percentile(np.abs(diff_map1), 99) # 用百分位数来设置色条范围，避免极端值影响
        im_diff1 = ax_diff1.imshow(diff_map1, cmap='coolwarm', aspect='auto', origin='lower', extent=extent, vmin=-vmax, vmax=vmax)
        ax_diff1.set_title("'Very Poor' vs 'Excellent' Attention Difference\n(Red: More attention in Poor, Blue: More attention in Excellent)", fontsize=18)
        fig.colorbar(im_diff1, ax=ax_diff1, label="Attention Difference")

    if 'Poor' in stats and 'Good' in stats:
        ax_diff2 = fig.add_subplot(gs[num_labels + 1, :])
        diff_map2 = stats['Poor']['mean'] - stats['Good']['mean']
        vmax = np.percentile(np.abs(diff_map2), 99)
        im_diff2 = ax_diff2.imshow(diff_map2, cmap='coolwarm', aspect='auto', origin='lower', extent=extent, vmin=-vmax, vmax=vmax)
        ax_diff2.set_title("'Poor' vs 'Good' Attention Difference", fontsize=18)
        ax_diff2.set_xlabel("Time (ms)")
        fig.colorbar(im_diff2, ax=ax_diff2, label="Attention Difference")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ==============================================================================
# 已有的绘图函数 (保持不变, 但现在从主函数调用)
# ==============================================================================

def plot_performance_summary(df, quality_labels, save_path, config):
    """绘制性能总结图，标题中包含阵列ID。"""
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f"Performance Summary for Array {config['array_id']}", fontsize=22)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1,1])
    ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[1, 1])
    colors = plt.cm.get_cmap('viridis', len(quality_labels))
    for i, quality in enumerate(quality_labels):
        subset = df[df['quality'] == quality]
        ax1.scatter(subset['true_csi'], subset['pred_csi'], color=colors(i), label=f"{quality} (n={len(subset)})", alpha=0.7, s=50)
    ax1.plot([0, 1], [0, 1], 'r--', lw=2.5, label='Perfect Prediction')
    ax1.set_xlabel('True CSI', fontsize=14); ax1.set_ylabel('Predicted CSI', fontsize=14)
    ax1.set_title('Predicted vs. True CSI by Cement Quality', fontsize=18)
    ax1.legend(); ax1.grid(True); ax1.set_aspect('equal', adjustable='box')
    quality_counts = df['quality'].value_counts().reindex(quality_labels)
    quality_counts.plot(kind='bar', ax=ax2, color=[colors(i) for i in range(len(quality_labels))])
    ax2.set_title('Sample Distribution in Validation Set', fontsize=16)
    ax2.set_ylabel('Number of Samples', fontsize=14); ax2.tick_params(axis='x', rotation=25)
    df['error'] = df['pred_csi'] - df['true_csi']
    for i, quality in enumerate(quality_labels):
        df[df['quality'] == quality]['error'].plot(kind='kde', ax=ax3, label=quality, color=colors(i), lw=2)
    ax3.set_title('Prediction Error Density by Quality', fontsize=16)
    ax3.set_xlabel('Prediction Error (Predicted - True)', fontsize=14)
    ax3.legend(); ax3.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_multidim_samples(df, x_val, waveforms, model, layer_name, save_path, config):
    """为每个类别绘制一个多维度样本分析图。"""
    array_id = config['array_id']
    physical_params = config['physical']
    quality_labels = df['quality'].cat.categories.tolist()
    
    fig, axes = plt.subplots(len(quality_labels), 3, figsize=(24, 7 * len(quality_labels)), squeeze=False)
    fig.suptitle(f'Multi-dimensional Sample Analysis for Array {array_id}', fontsize=24, y=1.0)
    
    freqs_khz = _calculate_cwt_frequencies_khz(config)
    time_ms = np.arange(waveforms.shape[1]) * (1 / physical_params['sampling_rate_hz']) * 1000
    
    for i, quality in enumerate(quality_labels):
        subset_df = df[df['quality'] == quality]
        if subset_df.empty:
            for j in range(3): axes[i, j].axis('off')
            continue
        
        sample = subset_df.sample(1, random_state=i)
        sample_idx_in_df = sample.index[0]
        original_data_idx = sample['original_index'].iloc[0]

        img_array = np.expand_dims(x_val[original_data_idx], axis=(0, -1))
        heatmap = make_gradcam_heatmap(img_array, model, layer_name).astype(np.float32)
        
        time_attention = gaussian_filter1d(np.mean(heatmap, axis=0), sigma=5) # 减小sigma以适应更短的时间轴
        freq_attention = gaussian_filter1d(np.mean(heatmap, axis=1), sigma=5)
        
        max_attention_time_idx = np.argmax(time_attention)
        max_attention_freq_idx = np.argmax(freq_attention)
        
        center_time_ms = time_ms[max_attention_time_idx]
        center_freq_khz = freqs_khz[max_attention_freq_idx]

        t_min, t_max = (center_time_ms - 0.5, center_time_ms + 0.5)
        
        title = f'Quality: {quality}\nTrue: {df.loc[sample_idx_in_df, "true_csi"]:.2f}, Pred: {df.loc[sample_idx_in_df, "pred_csi"]:.2f}'
        extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

        superimposed_img = superimpose_gradcam(x_val[original_data_idx], heatmap)
        axes[i, 0].imshow(superimposed_img, extent=extent, aspect='auto', origin='lower')
        axes[i, 0].set_title(f"Grad-CAM Heatmap\n{title}", fontsize=14); axes[i, 0].set_ylabel("Frequency (kHz)", fontsize=12)

        waveform_sample = waveforms[original_data_idx]
        axes[i, 1].plot(time_ms, waveform_sample)
        axes[i, 1].axvspan(t_min, t_max, color='orange', alpha=0.3, label=f'Attention @ {center_time_ms:.1f}ms')
        axes[i, 1].set_title(f"Original Waveform\n{title}", fontsize=14); axes[i, 1].legend()

        low_cut_hz = max(100, (center_freq_khz - 2.5) * 1000)
        high_cut_hz = (center_freq_khz + 2.5) * 1000
        
        sos = butter(4, [low_cut_hz, high_cut_hz], btype='band', fs=physical_params['sampling_rate_hz'], output='sos')
        filtered_waveform = sosfiltfilt(sos, waveform_sample)
        axes[i, 2].plot(time_ms, filtered_waveform, 'r')
        axes[i, 2].axvspan(t_min, t_max, color='orange', alpha=0.3)
        axes[i, 2].set_title(f"Filtered to Attention Band ({low_cut_hz/1000:.1f}-{high_cut_hz/1000:.1f} kHz)\n{title}", fontsize=14)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, 4); ax.grid(True)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.savefig(save_path, dpi=300); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为指定接收器阵列运行最终的模型性能评估和可视化分析。")
    parser.add_argument(
        '--array',
        type=str,
        default='03',
        help="指定要分析的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    args = parser.parse_args()

    config = get_config(args.array)
    run_analysis(config)