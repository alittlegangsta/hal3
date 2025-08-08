# src/interpretation/generate_analysis_candidates.py

import os
import sys
import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- 确保能找到 config 和其他模块 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import (
    load_config, SAMPLING_RATE_HZ
)
from src.interpretation.run_analysis import _calculate_cwt_frequencies_khz # 复用函数
from src.interpretation.grad_cam import make_gradcam_heatmap, superimpose_gradcam

def generate_candidates(num_per_category=3):
    """
    为每个人工分析的胶结类别挑选指定数量的候选样本，
    并生成它们的“身份证”（样本序号）和注意力图。
    """
    print("--- [开始生成分析候选样本] ---")
    
    # 1. 加载所需资源
    config = load_config()
    paths = config['paths']
    model_params = config['modeling']
    
    print("正在加载模型和数据...")
    model = load_model(paths['model_checkpoint'], compile=False)
    with h5py.File(paths['training_ready_data'], 'r') as hf:
        x_val = hf['x_val'][:]
        y_val = hf['y_val'][:]
    
    x_val_processed = np.expand_dims(x_val, axis=-1)
    y_pred = model.predict(x_val_processed, batch_size=model_params['batch_size']).flatten()
    
    df_val = pd.DataFrame({'true_csi': y_val, 'pred_csi': y_pred})
    bins = [c[1] for c in config['data_processing']['csi_bins']] + [1.0]
    labels = [c[0] for c in config['data_processing']['csi_bins']]
    df_val['quality'] = pd.cut(df_val['true_csi'], bins=bins, labels=labels, include_lowest=True, right=True)
    print("数据加载和预测完成。")

    # 2. 为每个类别挑选样本并作图
    output_dir = os.path.join(paths['plot_dir'], 'analysis_candidates')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n候选样本的图表将被保存在: {output_dir}")

    last_conv_layer_name = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    freqs_khz = _calculate_cwt_frequencies_khz()
    time_ms = np.arange(x_val.shape[2]) * (1 / SAMPLING_RATE_HZ) * 1000
    extent = [time_ms.min(), time_ms.max(), freqs_khz.min(), freqs_khz.max()]

    print("\n--- 候选样本信息 ---")
    for quality_level in labels:
        subset_df = df_val[df_val['quality'] == quality_level]
        if len(subset_df) < num_per_category:
            print(f"警告: '{quality_level}' 类别的样本不足 {num_per_category} 个，将使用所有可用样本。")
            sample_indices = subset_df.index.tolist()
        else:
            sample_indices = subset_df.sample(num_per_category, random_state=42).index.tolist()
        
        print(f"\n类别: '{quality_level}'")
        print(f"  - 挑选出的样本序号 (Validation Set Index): {sample_indices}")

        for idx in sample_indices:
            img_array = np.expand_dims(x_val[idx], axis=(0, -1))
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name).astype(np.float32)
            superimposed_img = superimpose_gradcam(x_val[idx], heatmap)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(superimposed_img, extent=extent, aspect='auto', origin='lower')
            title = (f"Candidate for Analysis - Category: {quality_level}\n"
                     f"Validation Set Index: {idx}\n"
                     f"True CSI: {df_val.loc[idx, 'true_csi']:.3f}, Predicted CSI: {df_val.loc[idx, 'pred_csi']:.3f}")
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Time (ms)", fontsize=12)
            ax.set_ylabel("Frequency (kHz)", fontsize=12)
            ax.set_xlim(0, 4)
            
            output_path = os.path.join(output_dir, f"candidate_{quality_level}_index_{idx}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    print("\n--- [候选样本生成完毕] ---")
    print("请检查上述输出的样本序号，并查看对应的图片进行人工分析。")

if __name__ == '__main__':
    generate_candidates()