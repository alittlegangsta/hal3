import sys
import os
import numpy as np
import h5py
import tensorflow as tf

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src import config
from src.modeling.train import get_normalization_stats
from src.modeling.dataset import create_dataset

def run_diagnostics():
    """
    运行数据管道的端到端诊断，以捕获潜在的数据污染问题。
    """
    print("="*60)
    print("🔬 Running Data Pipeline Diagnostics")
    print("="*60)

    # --- 步骤 1: 检查原始HDF5文件 ---
    print("\n--- Step 1: Checking raw scalogram data ---")
    try:
        with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
            scalograms_dset = hf['scalograms']
            print(f"  - Dataset shape: {scalograms_dset.shape}")
            
            # 随机抽查几个样本，检查是否存在NaN或inf
            num_samples_to_check = min(100, len(scalograms_dset))
            check_indices = np.random.choice(len(scalograms_dset), num_samples_to_check, replace=False)
            
            has_nan = False
            has_inf = False
            for i in check_indices:
                sample = scalograms_dset[i]
                if np.isnan(sample).any():
                    has_nan = True
                if np.isinf(sample).any():
                    has_inf = True
            
            if has_nan:
                print("  ❌ FATAL: NaN values found in the raw scalogram data!")
            else:
                print("  ✅ OK: No NaN values found in checked samples.")
            
            if has_inf:
                print("  ❌ FATAL: Infinite values found in the raw scalogram data!")
            else:
                print("  ✅ OK: No Infinite values found in checked samples.")
                
    except Exception as e:
        print(f"  ❌ FAILED to read or check HDF5 file: {e}")
        return

    # --- 步骤 2: 验证标准化统计量 ---
    print("\n--- Step 2: Verifying normalization statistics ---")
    try:
        # 使用一小部分训练样本来计算统计量
        with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
            total_samples = len(hf['csi_labels'])
        
        train_indices = np.random.choice(np.arange(total_samples), min(2000, total_samples), replace=False)
        mean, std = get_normalization_stats(config.SCALOGRAM_DATA_PATH, train_indices)

        if not np.isfinite(mean) or not np.isfinite(std):
            print(f"  ❌ FATAL: Calculated mean ({mean}) or std ({std}) is not a finite number!")
        elif std < 1e-6:
            print(f"  ⚠️ WARNING: Calculated std ({std}) is close to zero. Data might be constant.")
        else:
            print("  ✅ OK: Normalization stats are finite and seem reasonable.")

    except Exception as e:
        print(f"  ❌ FAILED to calculate normalization stats: {e}")
        return

    # --- 步骤 3: 检查最终送入模型的数据批次 ---
    print("\n--- Step 3: Inspecting the final data batch ---")
    try:
        batch_size = 32
        dataset = create_dataset(config.SCALOGRAM_DATA_PATH, train_indices, batch_size, mean, std)
        
        # 从数据集中取出一个批次
        x_batch, y_batch = next(iter(dataset))
        
        print(f"  - Batch shape (features): {x_batch.shape}")
        print(f"  - Batch shape (labels): {y_batch.shape}")
        
        x_batch_np = x_batch.numpy()

        if np.isnan(x_batch_np).any():
            print("  ❌ FATAL: NaN values found in the final batch sent to the model!")
        else:
            print("  ✅ OK: No NaN values found in the final batch.")
            
        if np.isinf(x_batch_np).any():
            print("  ❌ FATAL: Infinite values found in the final batch sent to the model!")
        else:
            print("  ✅ OK: No Infinite values found in the final batch.")

        print("\n--- Batch Data Statistics ---")
        print(f"  Min: {np.min(x_batch_np):.4f}")
        print(f"  Max: {np.max(x_batch_np):.4f}")
        print(f"  Mean: {np.mean(x_batch_np):.4f}")
        print(f"  Std Dev: {np.std(x_batch_np):.4f}")

    except Exception as e:
        print(f"  ❌ FAILED to create or inspect the data batch: {e}")
        return

    print("\n="*60)
    print("🔬 Diagnostics Complete.")
    print("="*60)

if __name__ == "__main__":
    run_diagnostics()