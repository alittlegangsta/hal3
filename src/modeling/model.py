import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_regressor(input_shape):
    """
    构建一个增强的CNN回归模型。

    Args:
        input_shape (tuple): 输入尺度图的形状 (height, width, channels).

    Returns:
        A Keras Model instance.
    """
    model_input = layers.Input(shape=input_shape)

    # --- 核心修改：移除了 layers.Rescaling(1./255.) ---
    # x = layers.Rescaling(1./255.)(model_input)  <- 删除这一行

    # 我们直接从原始输入开始
    x = model_input

    # 第1个卷积块
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 第2个卷积块
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 【推荐】增加第3个卷积块以增强学习能力
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # 回归头
    x = layers.Flatten()(x) # 使用Flatten代替GlobalAveragePooling2D以保留更多信息
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # 输出层
    # 使用 'sigmoid' 激活函数，因为它天然将输出限制在 (0, 1) 区间，
    # 这与我们的CSI定义完美匹配，可以帮助模型更快收敛。
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=model_input, outputs=output)
    
    return model