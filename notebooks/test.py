# Cell 1: 导入库和配置
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def find_project_root():
    """通过查找项目根目录的特征文件（如 requirements.txt）定位根路径"""
    current_dir = os.path.abspath(os.getcwd())  # 当前工作目录的绝对路径
    while True:
        # 检查当前目录是否包含项目根目录的特征文件（根据你的实际情况调整）
        if os.path.exists(os.path.join(current_dir, "requirements.txt")):
            return current_dir
        # 向上回溯一层目录
        parent_dir = os.path.dirname(current_dir)
        # 防止无限循环（到达系统根目录时退出）
        if parent_dir == current_dir:
            raise FileNotFoundError("未找到项目根目录（请确保项目包含 requirements.txt）")
        current_dir = parent_dir

# 获取项目根目录路径
project_root = find_project_root()

# 构造 src 目录的绝对路径并添加到 Python 搜索路径
src_dir = os.path.join(project_root, "src")
sys.path.append(src_dir)

# --------------------------
# 验证导入是否成功
# --------------------------
try:
    from config import ULTRASONIC_PATH, SONIC_PATH, ORIENTATION_PATH
    print("✅ 成功导入 config 模块！")
    # 打印路径验证（可选）
    print("ULTRASONIC_PATH:", ULTRASONIC_PATH)
    print("SONIC_PATH:", SONIC_PATH)
    print("ORIENTATION_PATH:", ORIENTATION_PATH)
except ImportError as e:
    print("❌ 导入 config 失败:", e)

# Cell 2: 定义一个辅助函数来探查.mat文件内容
def explore_mat_file(file_path):
    """加载并打印.mat文件的顶层结构和数据信息"""
    print(f"--- Exploring file: {os.path.basename(file_path)} ---")
    try:
        mat_data = scipy.io.loadmat(file_path, squeeze_me=True)
        print("Keys:", mat_data.keys())
        
        for key, value in mat_data.items():
            if key.startswith('__'):
                continue
            print(f"\n> Content of '{key}':")
            if isinstance(value, np.ndarray) and value.dtype.names:
                # 这是一个结构体数组
                print(f"  Type: Struct Array")
                print(f"  Fields: {value.dtype.names}")
                # 打印结构体中每个字段的形状
                for field in value.dtype.names:
                    # 访问结构体中的字段
                    field_data = value[field]
                    print(f"    - Field '{field}' shape: {field_data.shape}, dtype: {field_data.dtype}")
            elif isinstance(value, np.ndarray):
                 print(f"  Type: Array")
                 print(f"  Shape: {value.shape}, dtype: {value.dtype}")
            else:
                 print(f"  Type: {type(value)}")

    except Exception as e:
        print(f"  Error loading or exploring file: {e}")
    print("-" * 50)

# Cell 3: 探查所有原始数据文件
explore_mat_file(ULTRASONIC_PATH)
explore_mat_file(SONIC_PATH)
explore_mat_file(ORIENTATION_PATH)

# Cell 4: 加载并可视化一个样本波形
print("\n--- Visualizing a sample sonic waveform ---")
sonic_mat = scipy.io.loadmat(SONIC_PATH, squeeze_me=True)
# 根据策略文档，数据在 'XSILMR03' 结构体中
sonic_struct = sonic_mat['XSILMR03']
waveforms_A = sonic_struct['WaveRng03SideA'] # A方位接收器全波形
depths = sonic_struct['Depth']

# 绘制第一个深度点的波形
sample_waveform = waveforms_A[:, 0]
time_axis_us = np.arange(sample_waveform.shape[0]) * 10 # 采样间隔10us

plt.figure(figsize=(10, 5))
plt.plot(time_axis_us, sample_waveform)
plt.title('Sample Raw Waveform (Receiver A, First Depth)')
plt.xlabel('Time (us)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

print(f"Sonic data shape (Receiver A): {waveforms_A.shape}")
print(f"Sonic depth shape: {depths.shape}")