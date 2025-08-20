# src/interpretation/generate_analysis_candidates.py (最终版 - 垂直排列)

import os
import sys
import argparse
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- 确保能找到 config 和其他模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 动态导入配置函数和复用函数
from config import get_config
from src.interpretation.run_analysis import _calculate_cwt_frequencies_khz
from src.interpretation.grad_cam import make_gradcam_heatmap, superimpose_gradcam

def generate_candidates(config, num_per_category=3):
    """
    为每个类别挑选候选样本，并生成其注意力热力图与对应的高通滤波后波形图（垂直排列）。
    """
    array_id = config['array_id']
    print(f"--- [开始为阵列 {array_id} 生成分析候选样本] ---")
    print(f"--- [每个类别将生成 {num_per_category} 个样本] ---")
    
    # 1. 从配置中加载所需资源
    paths = config['paths']
    model_params = config['modeling']
    data_proc_params = config['data_processing']
    physical_params = config['physical']

    print("正在加载模型和数据...")
    required_files = {
        'model': paths['model_checkpoint'],
        'training_data': paths['training_ready_data'],
        'aligned_data': paths['aligned_data'],
        'split_indices': paths['split_indices']
    }
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"错误: {name} 文件未找到于 {path}")
            print("请确保已为当前阵列完整运行了所有前期步骤。")
            return
            
    model = load_model(required_files['model'])
    with h5py.File(required_files['training_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    
    aligned_data_file = h5py.File(required_files['aligned_data'], 'r')
    waveforms_dset = aligned_data_file['waveforms']
    split_indices = np.load(required_files['split_indices'])
    val_indices_map = split_indices['val_indices']

    x_val_processed = np.expand_dims(x_val, axis=-1)
    y_pred = model.predict(x_val_processed, batch_size=model_params['batch_size']).flatten()
    
    df_val = pd.DataFrame({'true_csi': y_val, 'pred_csi': y_pred})
    bins = [c[1] for c in data_proc_params['csi_bins']] + [1.0]
    labels = [c[0] for c in data_proc_params['csi_bins']]
    df_val['quality'] = pd.cut(df_val['true_csi'], bins=bins, labels=labels, include_lowest=True, right=True)
    print("数据加载和预测完成。")

    # 2. 为每个类别挑选样本并作图
    output_dir = os.path.join(paths['plot_dir'], 'analysis_candidates')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n候选样本的图表将被保存在: {output_dir}")

    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    
    freqs_khz = _calculate_cwt_frequencies_khz(config)
    time_ms = np.arange(physical_params['waveform_timesteps']) * (1 / physical_params['sampling_rate_hz']) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

    print(f"\n--- 阵列 {array_id} 的候选样本信息 ---")
    for quality_level in labels:
        subset_df = df_val[df_val['quality'] == quality_level]
        if len(subset_df) < num_per_category:
            print(f"警告: '{quality_level}' 类别的样本不足 {num_per_category} 个，将使用所有可用样本 ({len(subset_df)}个)。")
            sample_indices = subset_df.index.tolist()
        else:
            sample_indices = subset_df.sample(num_per_category, random_state=model_params['random_seed']).index.tolist()
        
        print(f"\n类别: '{quality_level}'")
        print(f"  - 挑选出的样本序号 (Validation Set Index): {sample_indices}")

        for idx in sample_indices:
            # --- 核心修改：修改子图布局和尺寸 ---
            fig, axes = plt.subplots(2, 1, figsize=(12, 12)) # 2行1列，高度加倍
            
            # --- 上方子图：注意力热力图 (axes[0]) ---
            img_array = np.expand_dims(x_val[idx], axis=(0, -1))
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name).astype(np.float32)
            superimposed_img = superimpose_gradcam(x_val[idx], heatmap)
            
            axes[0].imshow(superimposed_img, extent=extent, aspect='auto', origin='lower')
            axes[0].set_title("Attention Heatmap", fontsize=16)
            axes[0].set_ylabel("Frequency (kHz)", fontsize=12)
            axes[0].set_xlim(0, 4)
            
            # --- 下方子图：高通滤波波形图 (axes[1]) ---
            global_hdf5_index = val_indices_map[idx]
            waveform = waveforms_dset[global_hdf5_index]
            
            axes[1].plot(time_ms, waveform, color='royalblue')
            axes[1].set_title("High-Pass Filtered Waveform", fontsize=16)
            axes[1].set_xlabel("Time (ms)", fontsize=12)
            axes[1].set_ylabel("Amplitude", fontsize=12)
            axes[1].set_xlim(0, 4)
            axes[1].grid(True, linestyle='--', alpha=0.6)
            
            # 设置总标题
            main_title = (f"Candidate for Analysis - Array: {array_id} - Category: {quality_level}\n"
                          f"Validation Set Index: {idx} (Global Index: {global_hdf5_index})\n"
                          f"True CSI: {df_val.loc[idx, 'true_csi']:.3f}, Predicted CSI: {df_val.loc[idx, 'pred_csi']:.3f}")
            fig.suptitle(main_title, fontsize=18)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应总标题
            
            output_path = os.path.join(output_dir, f"candidate_{quality_level}_index_{idx}_comparison.png")
            plt.savefig(output_path, dpi=150)
            plt.close(fig)

    aligned_data_file.close()
    
    print(f"\n--- [阵列 {array_id} 的候选样本生成完毕] ---")
    print("请检查图片，进行对比分析。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为指定接收器阵列生成模型分析的候选样本，包含热力图和波形图。")
    parser.add_argument(
        '--array', type=str, default='03',
        help="指定要分析的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    parser.add_argument(
        '--num_samples', type=int, default=3,
        help="指定每个胶结质量类别要生成的候选样本数量。"
    )
    args = parser.parse_args()
    config = get_config(args.array)
    generate_candidates(config, num_per_category=args.num_samples)