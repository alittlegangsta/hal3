# src/modeling/model.py (更新版)

import tensorflow as tf
from tensorflow.keras import layers, models

def build_adaptive_cnn_model(input_shape, n_samples):
    """
    构建一个根据样本数量自适应调整复杂度的CNN模型。
    
    Args:
        input_shape (tuple): 输入尺度图的形状 (height, width)。
        n_samples (int): 用于训练的样本数量。

    Returns:
        tf.keras.Model: 编译前的Keras模型。
    """
    full_input_shape = (*input_shape, 1)
    model_input = layers.Input(shape=full_input_shape)

    # 根据样本数量动态选择模型架构
    if n_samples <= 5000:
        # 中型模型
        print(f"  🏗️ Building Standard CNN Architecture for {n_samples} samples...")
        # 块 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        # 块 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        # 聚合
        x = layers.GlobalAveragePooling2D()(x)
        # 全连接层
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)

    else: # n_samples > 5000
        # 大型模型
        print(f"  🏗️ Building Large CNN Architecture for {n_samples} samples...")
        # 块 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(model_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        # 块 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        # 块 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        # 全连接层
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

    # 输出层
    output = layers.Dense(1, activation='sigmoid', name='csi_output')(x)
    model = models.Model(inputs=model_input, outputs=output)
    return model