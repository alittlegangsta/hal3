# config.py

import os

# --- 基础路径配置 ---
# 项目的根目录 (这个脚本config.py本身就在根目录，所以这样定义是正确的)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据存放目录
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# 处理后数据存放目录
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# 输出目录（图表、报告等）
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# 确保所有输出目录都存在
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --- 文件路径配置 ---
# 原始数据文件
ULTRASONIC_PATH = os.path.join(RAW_DATA_DIR, 'CAST.mat')
SONIC_PATH = os.path.join(RAW_DATA_DIR, 'XSILMR', 'XSILMR03.mat')
ORIENTATION_PATH = os.path.join(RAW_DATA_DIR, 'D2_XSI_RelBearing_Inclination.mat')
TRAINING_READY_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '03_training_ready.h5')

# 阶段一输出：对齐好的波形和CSI标签
ALIGNED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '01_aligned_waveforms.h5')

# 阶段二输出：CWT尺度图
SCALOGRAM_H5_PATH = os.path.join(PROCESSED_DATA_DIR, '02_scalograms.h5') 

# 阶段三输出：数据集划分索引和归一化统计
SPLIT_INDICES_PATH = os.path.join(PROCESSED_DATA_DIR, 'split_indices.npz')
NORM_STATS_PATH = os.path.join(PROCESSED_DATA_DIR, 'norm_stats.json')

# 阶段四输出：训练好的模型
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'best_model.keras')

# 新增：阶段五输出：TFRecord 文件
TFRECORD_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.tfrecord')
TFRECORD_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, 'val.tfrecord')

# --- 数据处理参数 ---
DEPTH_RANGE = (2732, 4132)  # (ft)
DEPTH_RESOLUTION_FT = 0.14
SAMPLING_RATE_HZ = 100000  # 10 µs -> 100 kHz
FILTER_CUTOFF_HZ = 1000
FILTER_ORDER = 4
ZC_THRESHOLD = 2.5
VERTICAL_WINDOW_SIZE = 2.0  # (ft)

# --- CWT变换参数 ---
CWT_WAVELET_NAME = 'cmor1.5-1.0' # 复数Morlet小波
CWT_SCALES_NUM = 150
CWT_FREQ_RANGE_KHZ = (1, 30)
CWT_BATCH_SIZE = 2048 # CWT处理时的批处理大小，防止内存溢出

# --- 模型训练参数 ---
MODELING_CONFIG = {
    'input_shape': (CWT_SCALES_NUM, 1024, 1), # (频率尺度, 时间步, 通道)
    'validation_split': 0.2,
    'random_seed': 42,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 1e-4,
}

# --- 数据集划分参数 (用于分层抽样) ---
DATA_PROCESSING_CONFIG = {
    'csi_bins': [
        ('Excellent', 0.0, 0.2),
        ('Good', 0.2, 0.4),
        ('Poor', 0.4, 0.7),
        ('Very Poor', 0.7, 1.0)
    ]
}

def load_config():
    """一个简单的函数，用于在其他脚本中加载配置字典。"""
    # 这个函数现在可以被安全地从任何地方调用
    return {
        'paths': {
            'aligned_data': ALIGNED_DATA_PATH,
            'scalogram_h5': SCALOGRAM_H5_PATH,
            'split_indices': SPLIT_INDICES_PATH,
            'norm_stats': NORM_STATS_PATH,
            'model_checkpoint': MODEL_CHECKPOINT_PATH,
            'plot_dir': PLOT_DIR,
            'training_ready_data': TRAINING_READY_DATA_PATH,
            'tfrecord_train': TFRECORD_TRAIN_PATH,
            'tfrecord_val': TFRECORD_VAL_PATH,
        },
        'modeling': MODELING_CONFIG,
        'data_processing': DATA_PROCESSING_CONFIG
    }