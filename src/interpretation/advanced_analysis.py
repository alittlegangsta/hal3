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

# ... (文件顶部的CSI_BINS, CSI_LABELS, CLASS_COLORS 定义保持不变) ...
CSI_BINS = [0, 0.2, 0.4, 0.7, 1.0]
CSI_LABELS = ['Excellent (0-0.2)', 'Good (0.2-0.4)', 'Poor (0.4-0.7)', 'Very Poor (0.7-1.0)']
CLASS_COLORS = ['green', 'blue', 'orange', 'red']

def find_last_conv_layer(model):
    """动态寻找模型中最后一个卷积层的名称 (复用函数)"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Model does not contain any Conv2D layers.")

def assign_csi_class(csi_values):
    """根据CSI值分配胶结等级"""
    return np.digitize(csi_values, bins=CSI_BINS) - 1

# ... (plot_class_sample_analysis, plot_scatter_and_histograms, plot_mean_attention_maps 函数保持不变) ...
def plot_class_sample_analysis(model, last_conv_layer_name, sample_idx):
    # 此函数内容无需修改
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

    img_array = np.expand_dims(np.expand_dims(scalogram, axis=0), axis=-1)
    pred_csi = model.predict(img_array, verbose=0)[0][0]

    heatmap = get_grad_cam(model, img_array, last_conv_layer_name)
    
    peak_indices = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_freq_idx, peak_time_idx = peak_indices
    peak_freq_hz = frequencies_hz[peak_freq_idx]
    
    low_cut = peak_freq_hz * 0.9
    high_cut = peak_freq_hz * 1.1
    nyquist = 0.5 * config.SAMPLING_RATE_HZ
    if high_cut >= nyquist:
        high_cut = nyquist * 0.99
    
    b, a = butter(4, [low_cut, high_cut], btype='band', fs=config.SAMPLING_RATE_HZ)
    filtered_waveform = filtfilt(b, a, original_waveform)
    
    print(f"  - Peak attention at Freq: {peak_freq_hz/1000:.2f} kHz. Applying bandpass filter.")

    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:])
    
    time_axis_ms = np.arange(scalogram.shape[1]) * config.SAMPLING_INTERVAL_US / 1000
    freq_axis_khz = frequencies_hz / 1000
    ax1.imshow(scalogram, aspect='auto', cmap='viridis', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    ax1.imshow(heatmap, cmap='jet', alpha=0.5, aspect='auto', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    ax1.set_title(f'Grad-CAM (Sample {sample_idx})\nTrue: {true_csi:.2f}, Pred: {pred_csi:.2f}')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (kHz)')
    
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
    # 此函数内容无需修改
    print("\n--- Plotting Overall Performance Visualizations ---")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)
    
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
    
    ax2 = fig.add_subplot(gs[0, 1])
    counts = [np.sum(class_assignments == i) for i in range(len(CSI_LABELS))]
    ax2.bar(CSI_LABELS, counts, color=CLASS_COLORS)
    ax2.set_title('Sample Distribution by Cement Quality')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=30)

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

def plot_mean_attention_maps(model, last_conv_layer_name, h5_path, indices):
    # 此函数内容无需修改
    print("\n--- Calculating and Plotting Mean Attention Maps ---")
    with h5py.File(h5_path, 'r') as hf:
        num_scales, num_timesteps = hf['scalograms'].shape[1:]
    
    sum_heatmap = np.zeros((num_scales, num_timesteps), dtype=np.float64)
    sum_sq_heatmap = np.zeros((num_scales, num_timesteps), dtype=np.float64)
    
    for i in tqdm(indices, desc="Aggregating Grad-CAMs"):
        with h5py.File(h5_path, 'r') as hf:
            scalogram = hf['scalograms'][i]
        
        img_array = np.expand_dims(np.expand_dims(scalogram, axis=0), axis=-1)
        heatmap = get_grad_cam(model, img_array, last_conv_layer_name)
        
        sum_heatmap += heatmap
        sum_sq_heatmap += np.square(heatmap)
    
    N = len(indices)
    mean_heatmap = sum_heatmap / N
    var_heatmap = (sum_sq_heatmap / N) - np.square(mean_heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = axes[0].imshow(mean_heatmap, cmap='jet', aspect='auto')
    axes[0].set_title('Mean Attention Map (Across All Validation Samples)')
    axes[0].set_xlabel('Time (Time Steps)')
    axes[0].set_ylabel('Frequency (Scale Index)')
    fig.colorbar(im1, ax=axes[0])
    
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
    
    # --- 修正之处：在这里统一、动态地寻找一次层名称 ---
    try:
        last_conv_layer_name = find_last_conv_layer(model)
        print(f"Dynamically found target layer for Grad-CAM: '{last_conv_layer_name}'")
    except ValueError as e:
        print(f"Error: {e}")
        return
    # --- 修正结束 ---
    
    # 2. 加载数据并进行预测
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_samples = len(hf['csi_labels'])
        scalograms_dset = hf['scalograms']
        all_true_csi = hf['csi_labels'][:]
        
        # --- 修正之处 ---
        # 为预测过程定义一个合理的批次大小，不再依赖config文件
        prediction_batch_size = 64 
        print(f"  Performing predictions with batch size: {prediction_batch_size}")

        all_pred_csi = model.predict(scalograms_dset, batch_size=prediction_batch_size, verbose=1)
        all_pred_csi = all_pred_csi.flatten()

    # 3. 计算误差和等级
    all_errors = all_pred_csi - all_true_csi
    all_class_assignments = assign_csi_class(all_true_csi)
    
    # 4. 绘制散点图和直方图
    plot_scatter_and_histograms(all_true_csi, all_pred_csi, all_errors, all_class_assignments)
    
    # 5. 绘制平均注意力图
    # 注意：为了节省时间，可以只对验证集或一个子集进行分析
    if config.QUICK_TEST_MODE:
        # 在快测模式下，只分析一部分样本的注意力
        num_analysis_samples = min(num_samples, 500)
        analysis_indices = np.random.choice(np.arange(num_samples), num_analysis_samples, replace=False)
        print(f"  QUICK TEST MODE: Analyzing attention for {num_analysis_samples} random samples.")
    else:
        analysis_indices = np.arange(num_samples)

    plot_mean_attention_maps(model, last_conv_layer_name, config.SCALOGRAM_DATA_PATH, analysis_indices)
    
    # 6. 随机挑选几个样本进行深入分析
    print("\n--- Performing deep-dive analysis on selected samples ---")
    for i, label in enumerate(CSI_LABELS):
        class_indices = np.where(all_class_assignments == i)[0]
        if len(class_indices) > 0:
            sample_to_plot = np.random.choice(class_indices)
            # --- 修正之处：传入找到的层名称 ---
            plot_class_sample_analysis(model, last_conv_layer_name, sample_to_plot)
        else:
            print(f"No samples found for class: {label}")