# src/utils/find_lr.py
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取当前文件（find_lr.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 向上两级目录：从 hal3/src/utils → hal3/src → hal3（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# 将项目根目录添加到 Python 搜索路径
sys.path.append(project_root)

from src import config
from src.modeling.model import build_adaptive_cnn_model
from src.modeling.dataset import create_dataset
from src.modeling.train import get_normalization_stats
from src.utils.callbacks import LRFinder

def run_lr_finder():
    print("--- Running Learning Rate Finder ---")
    
    # 使用快速测试模式的数据量进行查找
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_total_samples = len(hf['csi_labels'])
    
    train_indices = np.random.choice(np.arange(num_total_samples), config.QUICK_TEST_SAMPLES, replace=False)
    n_train = len(train_indices)

    print(f"Using {n_train} samples for LR finding.")

    # 计算标准化统计量
    mean, std = get_normalization_stats(config.SCALOGRAM_DATA_PATH, train_indices)
    
    # 动态设置超参数
    epochs = 3 # 跑几个周期足够了
    batch_size = 32
    
    # 创建数据集
    train_dataset = create_dataset(config.SCALOGRAM_DATA_PATH, train_indices, batch_size, mean, std, is_training=True)

    # 构建模型
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        input_shape = hf['scalograms'].shape[1:]
    model = build_adaptive_cnn_model(input_shape, n_train)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error'
    )

    # 设置LR Finder回调
    steps_per_epoch = len(train_indices) // batch_size
    lr_finder = LRFinder(start_lr=1e-7, end_lr=1e-1, steps=steps_per_epoch * epochs)

    # 运行查找
    model.fit(train_dataset, epochs=epochs, callbacks=[lr_finder], verbose=1)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(lr_finder.lrs, lr_finder.losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    save_path = os.path.join(config.PLOT_DIR, 'lr_finder_plot.png')
    plt.savefig(save_path)
    print(f"\nLR Finder plot saved to {save_path}")
    plt.show()
    
    print("\n请观察上图，找到损失下降最陡峭区域的中间位置，该处的学习率即为理想的最大学习率（max_lr），通常在1e-4到1e-2之间。")

if __name__ == '__main__':
    run_lr_finder()