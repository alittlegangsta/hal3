import os

# --- 流程控制 (Workflow Control) ---
# 设置为 True 可使用少量样本快速运行整个流程，用于调试。
# 设置为 False 可使用全量数据进行正式训练。
QUICK_TEST_MODE = False
QUICK_TEST_SAMPLES = 2000  # 快速测试模式下使用的样本数量

# --- 基础路径定义 (保持不变) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# --- 原始数据路径 (保持不变) ---
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
ULTRASONIC_PATH = os.path.join(RAW_DATA_DIR, 'CAST.mat')
SONIC_PATH = os.path.join(RAW_DATA_DIR, 'XSILMR', 'XSILMR03.mat')
ORIENTATION_PATH = os.path.join(RAW_DATA_DIR, 'D2_XSI_RelBearing_Inclination.mat')

# --- 处理后数据路径 (保持不变) ---
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ALIGNED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '01_aligned_data.h5')
SCALOGRAM_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, '02_scalograms.h5')

# --- 结果输出路径 (保持不变) ---
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
GRAD_CAM_DIR = os.path.join(PLOT_DIR, 'grad_cam_outputs')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')

# 确保所有输出目录都存在
for path in [PROCESSED_DATA_DIR, MODEL_DIR, PLOT_DIR, GRAD_CAM_DIR, LOG_DIR]:
    os.makedirs(path, exist_ok=True)

# --- 数据分析范围 (保持不变) ---
DEPTH_RANGE = (2732, 4132)  # (ft)

# --- 声波数据预处理参数 (保持不变) ---
SAMPLING_INTERVAL_US = 10  # 微秒 (µs)
SAMPLING_RATE_HZ = 1e6 / SAMPLING_INTERVAL_US
FILTER_CUTOFF_HZ = 1000  # 高通滤波截止频率
FILTER_ORDER = 4         # 滤波器阶数

# --- 数据对齐与插值参数 (保持不变) ---
DEPTH_RESOLUTION_FT = 0.14

# --- CSI 计算参数 (保持不变) ---
ZC_THRESHOLD = 2.5  # 声阻抗阈值
VERTICAL_WINDOW_SIZE = 2.0 # 使用2.0ft的物理路径模式

# --- 数据集平衡参数 (保持不变) ---
BALANCE_CSI_THRESHOLD = 0.1
MAX_LOW_CSI_SAMPLES = 5000

# --- CWT 时频变换参数 (保持不变) ---
WAVELET_NAME = 'cmor1.5-1.0'
TARGET_FREQ_RANGE_KHZ = (1, 30)
NUM_SCALES = 150

# --- CNN 模型架构参考参数 (更新) ---
# 注意：这些参数是构建自适应模型的参考，实际架构由 src/modeling/model.py 决定
CONV_FILTERS = (64, 128, 256)      # 大型模型中每个卷积块的滤波器数量
DENSE_UNITS = (256, 128)           # 大型模型中全连接层的神经元数量
DROPOUT_RATE = 0.5                 # Dropout比率
OUTPUT_ACTIVATION = 'sigmoid'      # 输出层激活函数

# --- 模型训练参数 (已移除) ---
# BATCH_SIZE, LEARNING_RATE, EPOCHS, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE
# 等参数现在由 src/modeling/train.py 根据数据集大小动态设置，以实现最优训练策略。

# --- Grad-CAM 可解释性参数 (修正) ---
# TARGET_CONV_LAYER_NAME 已被移除，因为代码将自动检测
VISUALIZATION_TIME_RANGE_MS = (0, 4)
VISUALIZATION_FREQ_RANGE_KHZ = (0, 30)