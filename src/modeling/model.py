import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape, config):
    """
    构建用于预测CSI的定制化CNN回归模型。
    (策略文档 5.1.2)
    
    Args:
        input_shape (tuple): 输入尺度图的形状 (height, width)。
        config (module): 从config.py导入的配置模块。

    Returns:
        tf.keras.Model: 编译前的Keras模型。
    """
    # 输入层需要通道信息，所以是 (height, width, 1)
    full_input_shape = (*input_shape, 1)
    
    model_input = layers.Input(shape=full_input_shape)
    
    # 在模型最开始增加一个归一化层，将输入值缩放到0-1范围
    x = layers.Rescaling(1./255.)(model_input) # 假设尺度图最大值在255附近，这是一个常用标准

    # --- 卷积特征提取基座 ---
    # 块 1
    x = layers.Conv2D(config.CONV_FILTERS[0], config.KERNEL_SIZE, activation='relu', padding='same')(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 块 2
    x = layers.Conv2D(config.CONV_FILTERS[1], config.KERNEL_SIZE, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # --- 回归头 ---
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config.DENSE_UNITS, activation='relu')(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    
    # 输出层
    # 只有一个神经元，用于输出连续的CSI值
    # 激活函数根据配置选择 'sigmoid' 或 'linear'
    output = layers.Dense(1, activation=config.OUTPUT_ACTIVATION, name='csi_output')(x)
    
    model = models.Model(inputs=model_input, outputs=output)
    
    return model