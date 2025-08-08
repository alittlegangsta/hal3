# src/interpretation/grad_cam.py (最终视觉优化版)

import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    为给定的图像和模型生成Grad-CAM热力图。
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # 对于回归任务，直接使用输出神经元的值作为目标
        class_channel = preds[0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 归一化热力图，并加上一个小的epsilon防止除以零
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def superimpose_gradcam(img, heatmap, alpha=0.5):
    """
    将热力图平滑放大并叠加到原始尺度图上，以获得更好的视觉效果。
    """
    # 使用 zoom 进行平滑插值，使热力图与原图尺寸匹配
    heatmap_resized = zoom(heatmap, (img.shape[0] / heatmap.shape[0], img.shape[1] / heatmap.shape[1]))
    
    # 使用 'jet' colormap 将热力图转换为彩色
    heatmap_jet = cm.get_cmap("jet")(heatmap_resized)[..., :3]
    
    # 将原图（已标准化）转换为适合叠加的格式 (灰度 -> RGB)
    # 我们需要先将标准化的数据恢复到0-1范围以便可视化
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img_rgb = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)
    
    # 将热力图和原图进行alpha混合
    superimposed_img = heatmap_jet * alpha + img_rgb * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 1) # 保证数值在0-1之间
    
    return superimposed_img