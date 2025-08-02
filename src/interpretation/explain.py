import os
import sys
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.interpretation.grad_cam import get_grad_cam

def run_explanation(sample_index: int):
    """
    加载模型和数据，为一个样本生成并可视化Grad-CAM图。
    """
    print("\n--- Running Model Explanation (Grad-CAM) ---")
    
    # 1. 检查模型和数据文件是否存在
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"Error: Model file not found at {config.BEST_MODEL_PATH}. Please run training first.")
        return
    if not os.path.exists(config.SCALOGRAM_DATA_PATH):
        print(f"Error: Scalogram data file not found at {config.SCALOGRAM_DATA_PATH}. Please run transform first.")
        return

    # 2. 加载模型
    print(f"Loading best model from {config.BEST_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    # 确保最后一个卷积层的名称正确
    # 您可以在模型summary中找到它，这里我们用config中的值
    last_conv_layer_name = config.TARGET_CONV_LAYER_NAME 
    print(f"Using target layer for Grad-CAM: '{last_conv_layer_name}'")

    # 3. 加载单个数据样本
    print(f"Loading sample at index: {sample_index}")
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        if sample_index >= len(hf['csi_labels']):
            print(f"Error: Sample index {sample_index} is out of bounds. Dataset has {len(hf['csi_labels'])} samples.")
            return
        scalogram = hf['scalograms'][sample_index]
        true_csi = hf['csi_labels'][sample_index]
        frequencies_hz = hf['frequencies_hz'][:]
        
    # 4. 准备图像输入
    # 模型需要一个批次作为输入, (1, height, width, 1)
    img_array = np.expand_dims(scalogram, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # 5. 模型预测
    pred_csi = model.predict(img_array)[0][0]
    print(f"  - True CSI: {true_csi:.4f}")
    print(f"  - Predicted CSI: {pred_csi:.4f}")

    # 6. 生成并叠加Grad-CAM热力图
    # 窜槽证据图
    heatmap_channeling = get_grad_cam(model, img_array, last_conv_layer_name, dual_view=False)
    # 良好胶结证据图
    heatmap_good_bond = get_grad_cam(model, img_array, last_conv_layer_name, dual_view=True)
    
    # 7. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    time_axis_ms = np.arange(scalogram.shape[1]) * config.SAMPLING_INTERVAL_US / 1000
    freq_axis_khz = frequencies_hz / 1000

    # 原始图
    axes[0].imshow(scalogram, aspect='auto', cmap='viridis', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    axes[0].set_title(f'Original Scalogram\nTrue CSI: {true_csi:.2f}, Pred CSI: {pred_csi:.2f}')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Frequency (kHz)')

    # 窜槽证据图
    axes[1].imshow(scalogram, aspect='auto', cmap='viridis', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    axes[1].imshow(heatmap_channeling, cmap='jet', alpha=0.5, aspect='auto', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    axes[1].set_title('Channeling Evidence Map (Positive Influence)')
    axes[1].set_xlabel('Time (ms)')

    # 良好胶结证据图
    axes[2].imshow(scalogram, aspect='auto', cmap='viridis', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    axes[2].imshow(heatmap_good_bond, cmap='jet', alpha=0.5, aspect='auto', extent=[time_axis_ms[0], time_axis_ms[-1], freq_axis_khz[0], freq_axis_khz[-1]])
    axes[2].set_title('Good Bond Evidence Map (Negative Influence)')
    axes[2].set_xlabel('Time (ms)')

    plt.tight_layout()
    save_path = os.path.join(config.GRAD_CAM_DIR, f'sample_{sample_index}_gradcam.png')
    plt.savefig(save_path)
    print(f"Grad-CAM visualization saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    # 可从命令行接收样本索引
    if len(sys.argv) > 1:
        sample_idx = int(sys.argv[1])
    else:
        sample_idx = 100 # 默认解释第100个样本
    run_explanation(sample_idx)