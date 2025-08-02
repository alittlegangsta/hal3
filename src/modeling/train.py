# src/modeling/train.py (更新版)

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import h5py
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.modeling.model import build_adaptive_cnn_model # 导入新模型
from src.modeling.dataset import create_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.callbacks import OneCycleLR # --- 新增 ---

def stratified_sample(csi_labels, num_samples):
    """执行分层采样"""
    df = pd.DataFrame({'csi': csi_labels})
    bins = pd.cut(df['csi'], bins=[0, 0.2, 0.4, 0.7, 1.0], right=False, labels=False)
    
    # 按比例分配样本
    df_train, _ = train_test_split(
        df, 
        train_size=num_samples, 
        stratify=bins, 
        random_state=42
    )
    return sorted(df_train.index.tolist())

def get_normalization_stats(h5_path, train_indices):
    """计算训练集的log1p标准化统计量"""
    print("  Calculating normalization stats from training data...")
    with h5py.File(h5_path, 'r') as hf:
        scalograms = hf['scalograms']
        
        # 为了节约内存，分批计算
        sums = 0.0
        sum_sqs = 0.0
        total_pixels = 0
        
        for i in tqdm(train_indices, desc="  Stats Calculation"):
            data = np.log1p(scalograms[i,:,:])
            sums += data.sum()
            sum_sqs += np.sum(np.square(data))
            total_pixels += data.size
            
    mean = sums / total_pixels
    std = np.sqrt(sum_sqs / total_pixels - np.square(mean))
    
    print(f"  - Calculated Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std

def run_training():
    """执行完整的、经过优化的模型训练流程"""
    if not os.path.exists(config.SCALOGRAM_DATA_PATH):
        print(f"Error: Scalogram data file not found at {config.SCALOGRAM_DATA_PATH}")
        return
        
    print("\n--- Starting Optimized Model Training ---")

    # 1. 加载完整数据集信息并应用采样策略
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_total_samples = len(hf['csi_labels'])
        all_csi_labels = hf['csi_labels'][:]
        input_shape = hf['scalograms'].shape[1:]

    if config.QUICK_TEST_MODE:
        print(f"🚀 Running in QUICK TEST mode with {config.QUICK_TEST_SAMPLES} samples.")
        from sklearn.model_selection import train_test_split
        # 使用分层采样获取测试子集
        sample_indices, _ = train_test_split(np.arange(num_total_samples), 
                                             train_size=config.QUICK_TEST_SAMPLES,
                                             stratify=pd.cut(all_csi_labels, bins=4),
                                             random_state=42)
    else:
        print("💪 Running in FULL DATASET mode.")
        sample_indices = np.arange(num_total_samples)
    
    # 2. 划分训练/验证集
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(sample_indices, 
                                                  test_size=0.2, 
                                                  random_state=42,
                                                  stratify=pd.cut(all_csi_labels[sample_indices], bins=4))

    n_train = len(train_indices)
    n_val = len(val_indices)
    print(f"Total samples for this run: {n_train + n_val}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # 3. 计算标准化所需的均值和标准差 (仅使用训练集)
    mean, std = get_normalization_stats(config.SCALOGRAM_DATA_PATH, train_indices)
    
    # 4. 根据样本量动态设置超参数
    if n_train <= 5000:
        epochs = 30 # 可以适当增加周期，因为OneCycleLR能有效防止过拟合
        batch_size = 32
        patience = 8 # 早停耐心值
    else: # 大数据集
        epochs = 50
        batch_size = 64
        patience = 10
        
    # --- 新增：在这里设置从LR Finder找到的最佳学习率 ---
    # 根据经验，1e-3通常是一个很好的起点
    MAX_LR = 1e-3 
    print(f"  Using OneCycleLR with max_lr: {MAX_LR}")
    print(f"  Dynamically set hyperparameters: epochs={epochs}, batch_size={batch_size}")

    # 5. 创建数据集
    train_dataset = create_dataset(config.SCALOGRAM_DATA_PATH, train_indices, batch_size, mean, std, is_training=True)
    val_dataset = create_dataset(config.SCALOGRAM_DATA_PATH, val_indices, batch_size, mean, std, is_training=False)

    # 6. 构建并编译模型
    model = build_adaptive_cnn_model(input_shape, n_train)
    # 注意：OneCycleLR会管理学习率，所以这里Adam的lr参数不重要
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    model.summary()

    # 7. 设置回调函数 (使用OneCycleLR)
    steps_per_epoch = n_train // batch_size
    
    callbacks = [
        ModelCheckpoint(
            filepath=config.BEST_MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        # --- 使用新回调，替换掉ReduceLROnPlateau ---
        OneCycleLR(
            max_lr=MAX_LR,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )
    ]

    # 8. 开始训练
    print("\nStarting training...")
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )

    print("\n--- Optimized Training Finished ---")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")

if __name__ == '__main__':
    run_training()