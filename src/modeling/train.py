# src/modeling/train.py (最终可配置版 - 已裁剪)

import os
import sys
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from config import get_config
from src.modeling.model import build_cnn_regressor

# --- 核心修改：TFRecord解析函数现在需要知道正确的形状 ---
def _parse_tfr_element(element, config):
    """解析一个TFRecord样本并恢复其动态形状。"""
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(element, feature_description)
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = example['label']
    
    # 从配置中获取正确的形状
    input_shape = config['modeling']['input_shape']
    # input_shape is (scales, timesteps, channels), we need (scales, timesteps) for reshape
    reshape_dims = [input_shape[0], input_shape[1]] 
    
    feature = tf.reshape(feature, reshape_dims) 
    feature = tf.expand_dims(feature, axis=-1)
    return feature, label

def augment(feature, label):
    """应用SpecAugment数据增强。"""
    feature_4d = tf.expand_dims(feature, axis=0)
    feature_aug = tfa.image.random_cutout(
        images=feature_4d, mask_size=(10, 40), constant_values=0 # 可以适当减小掩码尺寸
    )
    feature_aug = tfa.image.random_cutout(
        images=feature_aug, mask_size=(40, 20), constant_values=0
    )
    return tf.squeeze(feature_aug, axis=0), label

def create_dataset_from_tfrecord(tfrecord_path, config, batch_size, shuffle=True, use_augmentation=False):
    """从TFRecord文件创建一个高效的tf.data.Dataset。"""
    dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048)
        
    # --- 核心修改：将config传递给解析函数 ---
    # 使用 lambda 函数来包装，以便传入额外的 config 参数
    parser_fn = lambda element: _parse_tfr_element(element, config)
    dataset = dataset.map(parser_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if use_augmentation:
        print("为当前数据集启用数据增强 (SpecAugment)。")
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def train_model(config):
    """主训练函数"""
    array_id = config['array_id']
    print(f"--- [开始为阵列 {array_id} 训练模型 (数据已裁剪)] ---")
    
    strategy = tf.distribute.MirroredStrategy()
    print(f'检测到 {strategy.num_replicas_in_sync} 个GPU设备。')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f'混合精度策略已设置为: {policy.name}')

    paths = config['paths']
    model_params = config['modeling']
    
    global_batch_size = model_params['batch_size'] * strategy.num_replicas_in_sync
    print(f"全局批处理大小 (Global Batch Size): {global_batch_size}")

    if not os.path.exists(paths['tfrecord_train']) or not os.path.exists(paths['tfrecord_val']):
        print(f"错误: 未找到TFRecord文件于 {os.path.dirname(paths['tfrecord_train'])}")
        print("请先为当前阵列运行 'tfrecord' 步骤。")
        return

    print("从TFRecord文件创建分布式数据管道...")
    # --- 核心修改：传递 config ---
    train_dataset = create_dataset_from_tfrecord(paths['tfrecord_train'], config, global_batch_size, use_augmentation=True)
    val_dataset = create_dataset_from_tfrecord(paths['tfrecord_val'], config, global_batch_size, shuffle=False)
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

    print(f"--- 开始为阵列 {array_id} 进行模型训练 ---")
    history = model.fit(
        train_dataset,
        epochs=model_params['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )
    print(f"--- 阵列 {array_id} 的模型训练完毕 ---")
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为指定接收器阵列训练CNN模型。")
    parser.add_argument(
        '--array', type=str, default='03',
        help="指定要训练的声波接收器阵列编号 (例如: '03', '07', '11')。"
    )
    args = parser.parse_args()

    config = get_config(args.array)
    train_model(config)