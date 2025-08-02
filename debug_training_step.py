import sys
import os
import numpy as np
import h5py
import tensorflow as tf

# 确保能导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src import config
from src.modeling.model import build_adaptive_cnn_model
from src.modeling.train import get_normalization_stats
from src.modeling.dataset import create_dataset

def inspect_tensor(tensor, name):
    """打印张量的详细统计信息"""
    if tensor is None:
        print(f"  - {name}: None")
        return
    
    # 检查是否是TensorFlow EagerTensor，如果是则转换为NumPy数组
    tensor_np = tensor.numpy() if hasattr(tensor, 'numpy') else np.array(tensor)
    
    print(f"  - {name}:")
    print(f"    - Shape: {tensor_np.shape}")
    print(f"    - Has NaN: {np.isnan(tensor_np).any()}")
    print(f"    - Has Inf: {np.isinf(tensor_np).any()}")
    print(f"    - Min: {np.min(tensor_np):.6f}, Max: {np.max(tensor_np):.6f}, Mean: {np.mean(tensor_np):.6f}")

def run_single_step_debug():
    """
    运行单步训练调试，深入检查前向传播和反向传播的每一步。
    """
    print("="*60)
    print("🔬 Running Single-Step Training Diagnostics")
    print("="*60)

    # --- 1. 准备一个数据批次 ---
    print("\n--- Step 1: Preparing a single data batch ---")
    with h5py.File(config.SCALOGRAM_DATA_PATH, 'r') as hf:
        num_total_samples = len(hf['csi_labels'])
        input_shape = hf['scalograms'].shape[1:]
    
    train_indices = np.arange(num_total_samples)
    mean, std = get_normalization_stats(config.SCALOGRAM_DATA_PATH, train_indices)
    dataset = create_dataset(config.SCALOGRAM_DATA_PATH, train_indices, 32, mean, std)
    x_batch, y_batch = next(iter(dataset))
    print("  ✅ A single batch of data has been prepared and normalized.")
    inspect_tensor(x_batch, "Input Batch (x_batch)")

    # --- 2. 构建模型并检查初始权重 ---
    print("\n--- Step 2: Building model and inspecting initial weights ---")
    model = build_adaptive_cnn_model(input_shape, num_total_samples)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss='mse')
    inspect_tensor(model.trainable_variables[0], "Initial weights of first Conv layer")

    # --- 3. 执行一次完整的前向和后向传播 ---
    print("\n--- Step 3: Performing a single forward and backward pass ---")
    
    # --- 关键修正：创建一个损失函数对象 ---
    loss_fn = tf.keras.losses.MeanSquaredError()
    # --- 修正结束 ---

    with tf.GradientTape() as tape:
        # --- 3a. 前向传播 ---
        print("  --- 3a. Forward Pass ---")
        y_pred = model(x_batch, training=True)
        inspect_tensor(y_pred, "Model Predictions (y_pred)")
        
        # --- 3b. 计算损失 ---
        print("\n  --- 3b. Loss Calculation ---")
        # --- 关键修正：使用损失函数对象来计算损失 ---
        loss = loss_fn(y_batch, y_pred)
        inspect_tensor(loss, "Calculated Loss")

    # --- 3c. 计算梯度 ---
    print("\n  --- 3c. Gradient Calculation (Backward Pass) ---")
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # --- 4. 深入检查梯度 ---
    print("\n--- Step 4: Inspecting Gradients ---")
    all_gradients_are_finite = True
    for i, grad in enumerate(gradients):
        layer_name = model.trainable_variables[i].name
        if grad is None:
            print(f"  - Gradient for '{layer_name}' is None.")
            continue
        
        grad_np = grad.numpy()
        has_nan = np.isnan(grad_np).any()
        has_inf = np.isinf(grad_np).any()
        
        if has_nan or has_inf:
            all_gradients_are_finite = False
            print(f"  ❌ FATAL: Gradient for '{layer_name}' contains NaN/Inf!")
        else:
            print(f"  ✅ OK: Gradient for '{layer_name}' is finite.")
            print(f"      - Min: {np.min(grad_np):.6f}, Max: {np.max(grad_np):.6f}, Mean: {np.mean(grad_np):.6f}")

    if all_gradients_are_finite:
        print("\n  ✅ ALL GRADIENTS ARE FINITE. The backward pass is numerically stable.")
    else:
        print("\n  ❌ GRADIENT CALCULATION FAILED. The training process is numerically unstable.")

    # --- 5. 模拟一次权重更新 ---
    print("\n--- Step 5: Simulating a weight update ---")
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    inspect_tensor(model.trainable_variables[0], "Weights of first Conv layer AFTER one update")

    print("\n="*60)
    print("🔬 Single-Step Diagnostics Complete.")
    print("="*60)

if __name__ == "__main__":
    run_single_step_debug()