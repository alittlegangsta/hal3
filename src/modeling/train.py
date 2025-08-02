import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.modeling.model import build_cnn_model
from src.modeling.dataset import create_dataset

def run_training():
    """
    执行完整的模型训练、验证和保存流程。
    """
    if not os.path.exists(config.SCALOGRAM_DATA_PATH):
        print(f"Error: Scalogram data file not found at {config.SCALOGRAM_DATA_PATH}")
        print("Please run the 'transform' stage first.")
        return
        
    print("\n--- Starting Model Training ---")

    # 1. 准备数据集索引
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_samples = len(hf['csi_labels'])
        input_shape = hf['scalograms'].shape[1:]
    
    indices = np.arange(num_samples)
    np.random.shuffle(indices) # 随机打乱索引
    
    split_point = int(num_samples * (1 - config.VALIDATION_SPLIT))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # 2. 创建训练和验证数据集
    train_dataset = create_dataset(config.SCALOGRAM_DATA_PATH, train_indices, config, is_training=True)
    val_dataset = create_dataset(config.SCALOGRAM_DATA_PATH, val_indices, config, is_training=False)

    # 3. 构建并编译模型
    print("Building and compiling model...")
    model = build_cnn_model(input_shape, config)
    
    # 使用Adam优化器和均方误差损失函数
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mean_squared_error',
        metrics=['mean_absolute_error'] # MAE更直观
    )
    model.summary()

    # 4. 设置回调函数
    # ModelCheckpoint: 只保存在验证集上性能最好的模型
    checkpoint_cb = ModelCheckpoint(
        filepath=config.BEST_MODEL_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    # EarlyStopping: 如果验证集损失在一定轮次内没有改善，则提前停止训练
    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min',
        restore_best_weights=True, # 训练结束后，模型权重将恢复为最佳那一次
        verbose=1
    )

    # 5. 开始训练
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    print("\n--- Training Finished ---")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

if __name__ == '__main__':
    run_training()