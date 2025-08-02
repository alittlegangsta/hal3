import numpy as np
import tensorflow as tf

def get_grad_cam(model, img_array, last_conv_layer_name, dual_view=False):
    """
    为回归模型计算Grad-CAM热力图。
    (修正版：增加了将热力图缩放回原始图像尺寸的步骤)
    
    Args:
        model (tf.keras.Model): 训练好的模型。
        img_array (np.ndarray): 单个输入图像 (尺度图), shape (1, H, W, 1)。
        last_conv_layer_name (str): 最后一个卷积层的名称。
        dual_view (bool): 是否为“良好胶结证据图”模式 (计算负梯度的影响)。

    Returns:
        np.ndarray: 生成的、与原图尺寸相同的热力图。
    """
    # 1. 构建一个新模型
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. 计算梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0]
        if dual_view:
             class_channel = -class_channel

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 3. 计算通道重要性权重
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 4. 计算热力图
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. 应用ReLU并归一化
    # 添加一个微小的epsilon防止除以零
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    
    # --- 新增的关键步骤：将热力图缩放到与原始图像相同的尺寸 ---
    # 获取原始图像的 H 和 W
    original_height = img_array.shape[1]
    original_width = img_array.shape[2]
    
    # tf.image.resize 需要一个4D张量 (batch, height, width, channels)
    heatmap_4d = tf.expand_dims(tf.expand_dims(heatmap, axis=0), axis=-1)
    
    # 使用双线性插值进行缩放
    resized_heatmap = tf.image.resize(heatmap_4d, [original_height, original_width])
    
    # 移除多余的维度
    resized_heatmap_squeezed = tf.squeeze(resized_heatmap)
    # --- 修正结束 ---

    return resized_heatmap_squeezed.numpy()