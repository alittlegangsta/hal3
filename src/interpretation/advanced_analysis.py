import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.interpretation.grad_cam import get_grad_cam

# 定义CSI区间和标签
CSI_BINS = [0, 0.2, 0.4, 0.7, 1.0]
CSI_LABELS = ['Excellent (0-0.2)', 'Good (0.2-0.4)', 'Poor (0.4-0.7)', 'Very Poor (0.7-1.0)']
CLASS_COLORS = ['green', 'blue', 'orange', 'red']


def assign_csi_class(csi_values):
    """根据CSI值分配胶结等级"""
    # 使用 np.digitize 高效地进行分箱
    # bins的长度比labels多1，返回的索引从1开始，所以-1使其从0开始
    return np.digitize(csi_values, bins=CSI_BINS) - 1

def plot_class_sample_analysis(model, sample_idx):
    """
    对单个样本进行深入分析：Grad-CAM, 原始波形, 注意力滤波后波形
    """
    print(f"\n--- Analyzing Sample Index: {sample_idx} ---")
    
    with h5py.File(config.ALIGNED_DATA_PATH, 'r') as hf_aligned, \
         h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf_scalo:
        
        if sample_idx >= len(hf_scalo['csi_labels']):
            print(f"Error: Sample index {sample_idx} is out of bounds.")
            return

        original_waveform = hf_aligned['waveforms'][sample_idx]
        scalogram = hf_scalo['scalograms'][sample_idx]
        true_csi = hf_scalo['csi_labels'][sample_idx]
        frequencies_hz = hf_scalo['frequencies_hz'][:]

    # 准备模型输入并预测
    img_array = np.expand_dims(np.expand_dims(scalogram, axis=0), axis=-1)
    pred_csi = model.predict(img_array)[0][0]

    # --- Grad-CAM 分析 ---
    heatmap = get_grad_cam(model, img_array, config.TARGET_CONV_LAYER_NAME)
    
    # --- 注意力滤波分析 ---
    # 找到注意力最高点的索引
    peak_indices = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_freq_idx, peak_time_idx = peak_indices
    
    # 获取对应的中心频率 (Hz)
    peak_freq_hz = frequencies_hz[peak_freq_idx]
    
    # 定义一个频带进行带通滤波 (例如，中心频率上下10%)
    low_cut = peak_freq_hz * 0.9
    high_cut = peak_freq_hz * 1.1
    
    # 设计带通滤波器
    # 确保截止频率不超过Nyquist频率
    nyquist = 0.5 * config.SAMPLING_RATE_HZ
    if high_cut >= nyquist:
        high_cut = nyquist * 0.99
    
    b, a = butter(4, [low_cut, high_cut], btype='band', fs=config.SAMPLING_RATE_HZ)
    filtered_waveform = filtfilt(b, a, original_waveform)
    
    print(f"  - Peak attention at Freq: {peak_freq_hz/1000:.2f} kHz. Applying bandpass filter.")

    # --- 绘图 ---
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # 绘制Grad-CAM图
    time_axis_ms = np.arange(scalogram.shape[1]) * config.SAMPLING_INTERVAL_US / 1000
    freq_axis_khz = frequencies_hz / 1000
    ax1.imshow(scalogram, aspect='auto', cmap='viridis', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    ax1.imshow(heatmap, cmap='jet', alpha=0.5, aspect='auto', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    ax1.set_title(f'Grad-CAM (Sample {sample_idx})\nTrue: {true_csi:.2f}, Pred: {pred_csi:.2f}')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (kHz)')
    
    # 绘制波形对比图
    ax2.plot(time_axis_ms, original_waveform, label='Original Waveform', color='gray', alpha=0.8)
    ax2.plot(time_axis_ms, filtered_waveform, label=f'Attention-Filtered Waveform\n({low_cut/1000:.1f}-{high_cut/1000:.1f} kHz)', color='red', linewidth=2)
    ax2.set_title('Waveform Analysis')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR, f'sample_{sample_idx}_analysis.png')
    plt.savefig(save_path)
    print(f"  - Full analysis plot saved to {save_path}")
    plt.show()

def plot_scatter_and_histograms(true_csi, pred_csi, errors, class_assignments):
    """
    绘制“预测 vs 真实”散点图 和 按等级划分的样本/误差直方图
    """
    print("\n--- Plotting Overall Performance Visualizations ---")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. 散点图
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(CSI_LABELS):
        mask = (class_assignments == i)
        ax1.scatter(true_csi[mask], pred_csi[mask], alpha=0.5, label=label, color=CLASS_COLORS[i])
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Prediction')
    ax1.set_xlabel('True CSI')
    ax1.set_ylabel('Predicted CSI')
    ax1.set_title('Predicted vs. True CSI')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 样本分布直方图
    ax2 = fig.add_subplot(gs[0, 1])
    counts = [np.sum(class_assignments == i) for i in range(len(CSI_LABELS))]
    ax2.bar(CSI_LABELS, counts, color=CLASS_COLORS)
    ax2.set_title('Sample Distribution by Cement Quality')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=30)

    # 3. 误差分布直方图
    ax3 = fig.add_subplot(gs[1, :])
    for i, label in enumerate(CSI_LABELS):
        mask = (class_assignments == i)
        ax3.hist(errors[mask], bins=30, alpha=0.7, label=label, color=CLASS_COLORS[i])
    ax3.set_xlabel('Prediction Error (Predicted - True)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Error Distribution by Cement Quality')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR, 'performance_analysis_summary.png')
    plt.savefig(save_path)
    print(f"  - Performance summary plot saved to {save_path}")
    plt.show()
    

def plot_mean_attention_maps(model, h5_path, indices):
    """
    计算并绘制所有指定样本的平均注意力和注意力方差图
    """
    print("\n--- Calculating and Plotting Mean Attention Maps ---")
    with h5py.File(h5_path, 'r') as hf:
        num_scales, num_timesteps = hf['scalograms'].shape[1:]
    
    sum_heatmap = np.zeros((num_scales, num_timesteps), dtype=np.float64)
    sum_sq_heatmap = np.zeros((num_scales, num_timesteps), dtype=np.float64)
    
    for i in tqdm(indices, desc="Aggregating Grad-CAMs"):
        with h5py.File(h5_path, 'r') as hf:
            scalogram = hf['scalograms'][i]
        
        img_array = np.expand_dims(np.expand_dims(scalogram, axis=0), axis=-1)
        heatmap = get_grad_cam(model, img_array, config.TARGET_CONV_LAYER_NAME)
        
        sum_heatmap += heatmap
        sum_sq_heatmap += np.square(heatmap)
    
    N = len(indices)
    mean_heatmap = sum_heatmap / N
    # 计算方差: Var(X) = E[X^2] - (E[X])^2
    var_heatmap = (sum_sq_heatmap / N) - np.square(mean_heatmap)
    
    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制平均注意力图
    im1 = axes[0].imshow(mean_heatmap, cmap='jet', aspect='auto')
    axes[0].set_title('Mean Attention Map (Across All Validation Samples)')
    axes[0].set_xlabel('Time (Time Steps)')
    axes[0].set_ylabel('Frequency (Scale Index)')
    fig.colorbar(im1, ax=axes[0])
    
    # 绘制注意力方差图
    im2 = axes[1].imshow(var_heatmap, cmap='magma', aspect='auto')
    axes[1].set_title('Attention Variance Map')
    axes[1].set_xlabel('Time (Time Steps)')
    axes[1].set_ylabel('Frequency (Scale Index)')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOT_DIR, 'mean_attention_maps.png')
    plt.savefig(save_path)
    print(f"  - Mean attention maps saved to {save_path}")
    plt.show()


def run_advanced_analysis():
    """执行所有高级分析的主函数"""
    print("--- Starting Advanced Model Analysis ---")
    
    # 1. 加载模型
    model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    
    # 2. 加载数据并进行预测
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_samples = len(hf['csi_labels'])
        scalograms_dset = hf['scalograms']
        all_true_csi = hf['csi_labels'][:]
        
        # 为了节约内存，分批次进行预测
        all_pred_csi = model.predict(scalograms_dset, batch_size=config.BATCH_SIZE, verbose=1)
        all_pred_csi = all_pred_csi.flatten()

    # 3. 计算误差和等级
    all_errors = all_pred_csi - all_true_csi
    all_class_assignments = assign_csi_class(all_true_csi)
    
    # 4. 绘制散点图和直方图
    plot_scatter_and_histograms(all_true_csi, all_pred_csi, all_errors, all_class_assignments)
    
    # 5. 绘制平均注意力图
    plot_mean_attention_maps(model, config.SCALOGRAM_DATA_PATH, np.arange(num_samples))
    
    # 6. 随机挑选几个样本进行深入分析
    print("\n--- Performing deep-dive analysis on selected samples ---")
    # 从每个类别中随机选择一个样本
    for i, label in enumerate(CSI_LABELS):
        class_indices = np.where(all_class_assignments == i)[0]
        if len(class_indices) > 0:
            sample_to_plot = np.random.choice(class_indices)
            plot_class_sample_analysis(model, sample_to_plot)
        else:
            print(f"No samples found for class: {label}")