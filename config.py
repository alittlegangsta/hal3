# config.py (最终可配置版 - 已裁剪)

import os

# --- 基础路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# --- 默认的、全局的原始数据路径 ---
ULTRASONIC_PATH = os.path.join(RAW_DATA_DIR, 'CAST.mat')
ORIENTATION_PATH = os.path.join(RAW_DATA_DIR, 'D2_XSI_RelBearing_Inclination.mat')

# --- 数据处理的共享物理参数 ---
PHYSICAL_PARAMS = {
    'depth_range': (2732, 4132),
    'depth_resolution_ft': 0.14,
    'sampling_rate_hz': 100000,
    'filter_cutoff_hz': 1000,
    'filter_order': 4,
    'zc_threshold': 2.5,
    'vertical_window_size': 2.0,
    'cwt_wavelet_name': 'cmor1.5-1.0',
    'cwt_scales_num': 150,
    'cwt_freq_range_khz': (1, 30),
    'cwt_batch_size': 2048,
    # --- 核心修改：定义新的时间步长 ---
    'waveform_timesteps': 400, # 对应 4ms (400 * 10us)
}

# --- 模型训练的共享参数 ---
MODELING_CONFIG = {
    # --- 核心修改：使用新的时间步长更新模型输入形状 ---
    'input_shape': (PHYSICAL_PARAMS['cwt_scales_num'], PHYSICAL_PARAMS['waveform_timesteps'], 1),
    'validation_split': 0.2,
    'random_seed': 42,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 1e-4,
}

# --- 数据集划分的共享参数 ---
DATA_PROCESSING_CONFIG = {
    'csi_bins': [
        ('Excellent', 0.0, 0.2),
        ('Good', 0.2, 0.4),
        ('Poor', 0.4, 0.7),
        ('Very Poor', 0.7, 1.0)
    ]
}

def get_config(array_id: str = '03'):
    """
    根据接收器阵列ID动态生成所有路径和配置。
    """
    if not isinstance(array_id, str) or not array_id.isdigit():
        raise ValueError(f"array_id 必须是代表数字的字符串, 例如 '03', '07', '11'。收到: {array_id}")

    ARRAY_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed', f'array_{array_id}')
    ARRAY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, f'array_{array_id}')
    PLOT_DIR = os.path.join(ARRAY_OUTPUT_DIR, 'plots')
    MODEL_DIR = os.path.join(ARRAY_OUTPUT_DIR, 'models')

    os.makedirs(ARRAY_PROCESSED_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    sonic_filename = f'XSILMR{array_id}.mat'
    sonic_path = os.path.join(RAW_DATA_DIR, 'XSILMR', sonic_filename)
    if not os.path.exists(sonic_path):
        print(f"警告: 声波数据文件未找到: {sonic_path}")

    config = {
        'array_id': array_id,
        'paths': {
            'ultrasonic': ULTRASONIC_PATH,
            'sonic': sonic_path,
            'orientation': ORIENTATION_PATH,
            'base_processed_dir': ARRAY_PROCESSED_DIR,
            'plot_dir': PLOT_DIR,
            'model_dir': MODEL_DIR,
            'aligned_data': os.path.join(ARRAY_PROCESSED_DIR, '01_aligned_waveforms.h5'),
            'scalogram_h5': os.path.join(ARRAY_PROCESSED_DIR, '02_scalograms.h5'),
            'split_indices': os.path.join(ARRAY_PROCESSED_DIR, 'split_indices.npz'),
            'norm_stats': os.path.join(ARRAY_PROCESSED_DIR, 'norm_stats.json'),
            'training_ready_data': os.path.join(ARRAY_PROCESSED_DIR, '03_training_ready.h5'),
            'tfrecord_train': os.path.join(ARRAY_PROCESSED_DIR, 'train.tfrecord'),
            'tfrecord_val': os.path.join(ARRAY_PROCESSED_DIR, 'val.tfrecord'),
            'model_checkpoint': os.path.join(MODEL_DIR, 'best_model.keras'),
        },
        'modeling': MODELING_CONFIG,
        'data_processing': DATA_PROCESSING_CONFIG,
        'physical': PHYSICAL_PARAMS,
        'sonic_struct_key': f'XSILMR{array_id}'
    }
    return config

def load_config():
    print("警告: 调用了不带参数的旧版 load_config()。将使用默认的 array_id='03'。")
    return get_config('03')