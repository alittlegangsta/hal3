# src/modeling/train.py (最终修正版)

import os
import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import load_config
from src.modeling.model import build_cnn_regressor

# --- TFRecord解析函数 (保持不变) ---
def _parse_tfr_element(element):
    """解析一个TFRecord样本并恢复其形状。"""
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(element, feature_description)
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = example['label']
    
    feature = tf.reshape(feature, [150, 1024]) 
    feature = tf.expand_dims(feature, axis=-1)
    return feature, label

# --- 新增：数据增强函数 (修正版) ---
def augment(feature, label):
    """
    应用SpecAugment数据增强，并确保张量形状正确。
    """
    # feature的输入形状是 [150, 1024, 1] (H, W, C)
    # tfa.image.random_cutout 需要一个4D张量 [B, H, W, C]
    # 我们为它添加一个临时的批次维度 (B=1)
    feature_4d = tf.expand_dims(feature, axis=0)
    
    # 1. 随机频率掩码
    feature_aug = tfa.image.random_cutout(
        images=feature_4d,
        mask_size=(10, 80),
        constant_values=0
    )
    
    # 2. 随机时间掩码
    feature_aug = tfa.image.random_cutout(
        images=feature_aug, # 上一步的输出已经是4D，可以直接使用
        mask_size=(80, 20),
        constant_values=0
    )

    # 移除临时的批次维度，恢复为 [150, 1024, 1] 并返回
    return tf.squeeze(feature_aug, axis=0), label

def create_dataset_from_tfrecord(tfrecord_path, batch_size, shuffle=True, use_augmentation=False):
    """从TFRecord文件创建一个高效的tf.data.Dataset。"""
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)
        
    dataset = dataset.map(_parse_tfr_element, num_parallel_calls=tf.data.AUTOTUNE)
    
    if use_augmentation:
        print("为当前数据集启用数据增强 (SpecAugment)。")
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def train_model():
    """主训练函数 - 集成了数据增强、混合精度和多GPU策略。"""
    strategy = tf.distribute.MirroredStrategy()
    print(f'检测到 {strategy.num_replicas_in_sync} 个GPU设备。')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f'混合精度策略已设置为: {policy.name}')

    config = load_config()
    paths = config['paths']
    model_params = config['modeling']
    
    global_batch_size = model_params['batch_size'] * strategy.num_replicas_in_sync
    print(f"全局批处理大小 (Global Batch Size): {global_batch_size}")

    if not os.path.exists(paths['tfrecord_train']) or not os.path.exists(paths['tfrecord_val']):
        print("错误: 未找到TFRecord文件。请先运行 'tfrecord' 步骤。")
        return

    print("从TFRecord文件创建分布式数据管道...")
    train_dataset = create_dataset_from_tfrecord(paths['tfrecord_train'], global_batch_size, use_augmentation=True)
    val_dataset = create_dataset_from_tfrecord(paths['tfrecord_val'], global_batch_size, shuffle=False)
    print("数据管道创建成功。")

    with strategy.scope():
        model = build_cnn_regressor(model_params['input_shape'])
        loss_fn = tf.keras.losses.Huber()
        optimizer = Adam(learning_rate=model_params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mean_absolute_error']
        )
    model.summary()

    callbacks = [
        ModelCheckpoint(filepath=paths['model_checkpoint'], save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, mode='min', verbose=1)
    ]

    print("--- 开始模型训练 (使用数据增强、多GPU和混合精度) ---")
    history = model.fit(
        train_dataset,
        epochs=model_params['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print("--- 模型训练完毕 ---")
    return history