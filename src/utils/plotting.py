import matplotlib.pyplot as plt
import numpy as np

def plot_csi_distribution(csi_values, save_path=None):
    """
    绘制CSI值的分布直方图。
    
    Args:
        csi_values (np.ndarray): CSI值数组。
        save_path (str, optional): 图像保存路径. Defaults to None.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(csi_values, bins=50, edgecolor='black')
    plt.title('Distribution of Channeling Severity Index (CSI)')
    plt.xlabel('CSI Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"CSI distribution plot saved to {save_path}")
    plt.show()


def plot_grad_cam(scalogram, heatmap, time_axis, freq_axis, title, save_path=None):
    """
    将Grad-CAM热力图叠加在原始尺度图上进行可视化。
    
    Args:
        scalogram (np.ndarray): 2D尺度图。
        heatmap (np.ndarray): 2D Grad-CAM热力图。
        time_axis (np.ndarray): 时间轴 (ms)。
        freq_axis (np.ndarray): 频率轴 (kHz)。
        title (str): 图像标题。
        save_path (str, optional): 图像保存路径. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制原始尺度图
    im = ax.imshow(scalogram, aspect='auto', cmap='viridis', 
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    
    # 叠加热力图
    ax.imshow(heatmap, cmap='jet', alpha=0.5, aspect='auto',
              extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency (kHz)', fontsize=12)
    
    # 添加色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Amplitude')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grad-CAM plot saved to {save_path}")
    plt.show()