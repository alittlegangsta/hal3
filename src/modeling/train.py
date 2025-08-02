# src/modeling/train.py

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 添加根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import load_config
from src.modeling.dataset import DataGenerator
from src.modeling.model import build_cnn_regressor

def train_model():
    """主训练函数。"""
    # 1. 加载配置、数据索引和标准化统计数据
    config = load_config()
    paths = config['paths']
    model_params = config['modeling']
    
    with open(paths['norm_stats'], 'r') as f:
        norm_stats = json.load(f)
        
    indices_data = np.load(paths['split_indices'])
    train_indices = indices_data['train_indices']
    val_indices = indices_data['val_indices']
    
    with h5py.File(paths['aligned_data'], 'r') as hf:
        all_csi = hf['csi_labels'][:]

    # 2. 创建数据集生成器
    train_generator = DataGenerator(
        h5_path=paths['scalogram_h5'],
        indices=train_indices,
        csi_labels=all_csi,
        batch_size=model_params['batch_size'],
        norm_stats=norm_stats
    )
    
    val_generator = DataGenerator(
        h5_path=paths['scalogram_h5'],
        indices=val_indices,
        csi_labels=all_csi,
        batch_size=model_params['batch_size'],
        norm_stats=norm_stats,
        shuffle=False # 验证集不需要打乱
    )

    # 3. 创建 tf.data.Dataset 以获得最佳性能
    input_shape = model_params['input_shape']
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # 4. 构建并编译模型
    model = build_cnn_regressor(input_shape)
    
    # 使用Huber损失函数，它对异常值不那么敏感，更适合回归任务
    loss_fn = tf.keras.losses.Huber()
    
    model.compile(
        optimizer=Adam(learning_rate=model_params['learning_rate']),
        loss=loss_fn,
        metrics=['mean_absolute_error'] # MAE比MSE更直观
    )
    
    model.summary()

    # 5. 设置回调函数
    callbacks = [
        ModelCheckpoint(
            filepath=paths['model_checkpoint'],
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15, # 如果验证集损失在15个epoch内没有改善，则停止
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2, # 学习率衰减因子
            patience=5,  # 5个epoch不下降就降低学习率
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
    ]

    # 6. 开始训练
    print("--- Starting Model Training ---")
    history = model.fit(
        train_dataset,
        epochs=model_params['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("--- Model Training Finished ---")

    return history