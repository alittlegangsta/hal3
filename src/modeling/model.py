# src/modeling/model.py (最终性能版: SE-ResNet)

import tensorflow as tf
from tensorflow.keras import layers, models

def se_block(input_tensor, ratio=8):
    """
    Squeeze-and-Excitation (SE) 注意力模块。
    """
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return layers.multiply([input_tensor, se])

def resnet_block(input_tensor, filters, kernel_size=(3, 3)):
    """
    带有SE注意力机制的残差模块 (SE-ResNet Block)。
    """
    # 主路径
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 引入SE模块
    x = se_block(x)
    
    # 短路连接 (Residual Connection)
    # 如果输入和输出的通道数不同，则需要一个1x1卷积来匹配维度
    if input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
    else:
        shortcut = input_tensor
        
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_cnn_regressor(input_shape):
    """
    构建一个基于SE-ResNet的、能力更强的最终版CNN回归模型。
    """
    model_input = layers.Input(shape=input_shape)

    # 初始卷积层，用于初步特征提取和降采样
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # 堆叠多个SE-ResNet模块
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128) # 增加通道数，学习更复杂的特征
    x = layers.MaxPooling2D((2, 2))(x) # 进一步降采样

    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    
    # 回归头
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # 输出层
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=model_input, outputs=output)
    
    return model