# src/interpretation/run_analysis.py (终极修正版)

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
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pywt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import (
    load_config, SAMPLING_RATE_HZ, CWT_WAVELET_NAME, 
    CWT_SCALES_NUM, CWT_FREQ_RANGE_KHZ, DATA_PROCESSING_CONFIG
)
from src.interpretation.grad_cam import make_gradcam_heatmap, superimpose_gradcam

def _calculate_cwt_frequencies_khz():
    f_min_hz, f_max_hz = CWT_FREQ_RANGE_KHZ[0] * 1000, CWT_FREQ_RANGE_KHZ[1] * 1000
    central_freq = pywt.central_frequency(CWT_WAVELET_NAME, precision=8)
    scale_max = central_freq * SAMPLING_RATE_HZ / f_min_hz
    scale_min = central_freq * SAMPLING_RATE_HZ / f_max_hz
    scales = np.geomspace(scale_min, scale_max, CWT_SCALES_NUM)
    frequencies_hz = pywt.scale2frequency(CWT_WAVELET_NAME, scales) * SAMPLING_RATE_HZ
    return frequencies_hz / 1000

def run_analysis():
    print("--- [开始最终模型分析] ---")
    
    # 加载
    config = load_config()
    paths = config['paths']
    model_params = config['modeling']
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

    # 预测
    y_pred = model.predict(x_val_processed, batch_size=model_params['batch_size']).flatten()
    df_val = pd.DataFrame({'true_csi': y_val, 'pred_csi': y_pred, 'original_index': np.arange(len(y_val))})
    bins = [c[1] for c in DATA_PROCESSING_CONFIG['csi_bins']] + [1.0]
    labels = [c[0] for c in DATA_PROCESSING_CONFIG['csi_bins']]
    df_val['quality'] = pd.cut(df_val['true_csi'], bins=bins, labels=labels, include_lowest=True, right=True)

    # 绘图
    output_summary_path = os.path.join(paths['plot_dir'], 'performance_summary_plots.png')
    plot_performance_summary(df_val, labels, output_summary_path)
    print(f"  - 性能汇总图已保存至: {output_summary_path}")

    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    
    attention_maps_path = os.path.join(paths['plot_dir'], 'attention_mean_variance.png')
    mean_map, var_map = calculate_mean_attention_maps(model, x_val_processed, last_conv_layer_name)
    plot_attention_maps(mean_map, var_map, attention_maps_path)
    print(f"  - 平均注意力图已保存至: {attention_maps_path}")
    
    multi_dim_plot_path = os.path.join(paths['plot_dir'], 'multi_dimension_sample_analysis.png')
    plot_multidim_samples(df_val, x_val, original_waveforms_val, model, last_conv_layer_name, multi_dim_plot_path)
    print(f"  - 多维度样本对比图已保存至: {multi_dim_plot_path}")

    print("\n--- [分析成功完成] ---")

def plot_performance_summary(df, quality_labels, save_path):
    fig = plt.figure(figsize=(18, 16)); gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1,1])
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
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def calculate_mean_attention_maps(model, data, layer_name):
    all_heatmaps = []; subset_indices = np.random.choice(len(data), min(1000, len(data)), replace=False)
    for i in tqdm(subset_indices, desc="Calculating Grad-CAM for Attention Maps"):
        img_array = data[i:i+1]
        heatmap = make_gradcam_heatmap(img_array, model, layer_name)
        all_heatmaps.append(heatmap.astype(np.float32))
    all_heatmaps = np.array(all_heatmaps)
    return np.mean(all_heatmaps, axis=0), np.var(all_heatmaps, axis=0)

def plot_attention_maps(mean_map, var_map, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.15]})
    freqs_khz = _calculate_cwt_frequencies_khz()
    time_ms = np.arange(mean_map.shape[1]) * (1 / SAMPLING_RATE_HZ) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]
    
    im = axes[0].imshow(mean_map, cmap='jet', aspect='auto', origin='lower', extent=extent)
    axes[0].set_title('Mean Attention Map', fontsize=16); axes[0].set_ylabel('Frequency (kHz)'); axes[0].set_xlabel('Time (ms)')
    fig.colorbar(im, ax=axes[0], label='Mean Attention')
    
    vmax = np.percentile(var_map, 99.5)
    im = axes[1].imshow(var_map, cmap='plasma', aspect='auto', origin='lower', extent=extent, vmax=vmax)
    axes[1].set_title('Attention Variance Map', fontsize=16); axes[1].set_xlabel('Time (ms)')
    axes[1].text(0.95, 0.95, 'Note: Low variance indicates a\nconsistent attention strategy.', transform=axes[1].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.colorbar(im, ax=axes[1], label='Attention Variance')
    
    # 修正：移除不必要的X轴限制
    # for ax in axes:
    #     ax.set_xlim(0, 4)

    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close(fig)

def plot_multidim_samples(df, x_val, waveforms, model, layer_name, save_path):
    quality_labels = df['quality'].cat.categories.tolist(); fig, axes = plt.subplots(len(quality_labels), 3, figsize=(24, 7 * len(quality_labels)))
    fig.suptitle('Multi-dimensional Sample Analysis', fontsize=24, y=1.0)
    freqs_khz = _calculate_cwt_frequencies_khz(); time_ms = np.arange(waveforms.shape[1]) * (1 / SAMPLING_RATE_HZ) * 1000
    
    for i, quality in enumerate(quality_labels):
        subset_df = df[df['quality'] == quality]
        if subset_df.empty:
            for j in range(3): axes[i, j].axis('off')
            continue
        
        sample = subset_df.sample(1, random_state=i) # 使用不同的随机种子以获得不同的样本
        sample_idx_in_df = sample.index[0]
        original_data_idx = sample['original_index'].iloc[0]

        img_array = np.expand_dims(x_val[original_data_idx], axis=(0, -1))
        heatmap = make_gradcam_heatmap(img_array, model, layer_name).astype(np.float32)
        
        # --- 核心修正：使用高斯平滑来稳健地寻找注意力峰值 ---
        time_attention = gaussian_filter1d(np.mean(heatmap, axis=0), sigma=20) # 增大sigma以获得更平滑的曲线
        freq_attention = gaussian_filter1d(np.mean(heatmap, axis=1), sigma=5)
        
        max_attention_time_idx = np.argmax(time_attention)
        max_attention_freq_idx = np.argmax(freq_attention)
        
        center_time_ms = time_ms[max_attention_time_idx]
        center_freq_khz = freqs_khz[max_attention_freq_idx]

        time_window_ms = 1.0
        t_min, t_max = (center_time_ms - time_window_ms / 2, center_time_ms + time_window_ms / 2)
        
        title = f'Quality: {quality}\nTrue: {df.loc[sample_idx_in_df, "true_csi"]:.2f}, Pred: {df.loc[sample_idx_in_df, "pred_csi"]:.2f}'
        extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

        superimposed_img = superimpose_gradcam(x_val[original_data_idx], heatmap)
        axes[i, 0].imshow(superimposed_img, extent=extent, aspect='auto', origin='lower')
        axes[i, 0].set_title(f"Grad-CAM Heatmap\n{title}", fontsize=14); axes[i, 0].set_ylabel("Frequency (kHz)", fontsize=12)

        waveform_sample = waveforms[original_data_idx]
        axes[i, 1].plot(time_ms, waveform_sample)
        axes[i, 1].axvspan(t_min, t_max, color='orange', alpha=0.3, label=f'Attention @ {center_time_ms:.1f}ms')
        axes[i, 1].set_title(f"Original Waveform\n{title}", fontsize=14); axes[i, 1].legend()

        band_width_khz = 5.0
        low_cut_hz = max(100, (center_freq_khz - band_width_khz / 2) * 1000)
        high_cut_hz = (center_freq_khz + band_width_khz / 2) * 1000
        
        sos = butter(4, [low_cut_hz, high_cut_hz], btype='band', fs=SAMPLING_RATE_HZ, output='sos')
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
    run_analysis()