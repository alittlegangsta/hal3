import os

# --- 基础路径定义 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# --- 原始数据路径 ---
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
ULTRASONIC_PATH = os.path.join(RAW_DATA_DIR, 'CAST.mat')
SONIC_PATH = os.path.join(RAW_DATA_DIR, 'XSILMR', 'XSILMR03.mat')
ORIENTATION_PATH = os.path.join(RAW_DATA_DIR, 'D2_XSI_RelBearing_Inclination.mat')

# --- 处理后数据路径 ---
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ALIGNED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '01_aligned_data.h5')
SCALOGRAM_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '02_scalograms.h5')

# --- 结果输出路径 ---
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
GRAD_CAM_DIR = os.path.join(PLOT_DIR, 'grad_cam_outputs')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')

# 确保所有输出目录都存在
for path in [PROCESSED_DATA_DIR, MODEL_DIR, PLOT_DIR, GRAD_CAM_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 数据分析范围 ---
DEPTH_RANGE = (2732, 4132) # (ft)

# --- 声波数据预处理参数 (策略文档 1.3) ---
SAMPLING_INTERVAL_US = 10  # 微秒 (µs)
SAMPLING_RATE_HZ = 1e6 / SAMPLING_INTERVAL_US
FILTER_CUTOFF_HZ = 1000  # 高通滤波截止频率
FILTER_ORDER = 4         # 滤波器阶数

# --- 数据对齐与插值参数 (策略文档 2.2) ---
# 使用超声数据的分辨率作为参考
DEPTH_RESOLUTION_FT = 0.14

# --- CSI 计算参数 (策略文档 3.2) ---
ZC_THRESHOLD = 2.5  # 声阻抗阈值，低于此值为窜槽
# 垂直聚合窗口大小 (ft)，可选项: 0.5 (高分), 2.0 (物理路径)
VERTICAL_WINDOW_SIZE = 0.5

# --- 数据集平衡参数 (补充信息 #4) ---
# 平衡CSI值小于该阈值的样本数量
BALANCE_CSI_THRESHOLD = 0.1
# 保留的低CSI样本的最大数量，None表示不削减
MAX_LOW_CSI_SAMPLES = 5000 

# --- CWT 时频变换参数 (策略文档 4.2) ---
WAVELET_NAME = 'cmor1.5-1.0'  # Complex Morlet 小波
TARGET_FREQ_RANGE_KHZ = (1, 30) # 目标频段 (kHz)
NUM_SCALES = 150 # 尺度数量

# --- CNN 模型架构参数 (策略文档 5.1 & 附录) ---
INPUT_SHAPE = (NUM_SCALES, 1024) # (num_scales, time_steps) - time_steps可能会在预处理后确认
CONV_FILTERS = (32, 64)
KERNEL_SIZE = (3, 3)
DENSE_UNITS = 64
DROPOUT_RATE = 0.5
# 输出层激活函数: 'sigmoid' 或 'linear'
OUTPUT_ACTIVATION = 'sigmoid'

# --- 模型训练参数 (策略文档 5.2 & 附录) ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# --- Grad-CAM 可解释性参数 (策略文档 6) ---
# 通常是最后一个卷积层
TARGET_CONV_LAYER_NAME = 'conv2d_1' # 注意：这个名字需要根据实际创建的模型来调整
VISUALIZATION_TIME_RANGE_MS = (0, 4) # (ms)
VISUALIZATION_FREQ_RANGE_KHZ = (0, 30) # (kHz)