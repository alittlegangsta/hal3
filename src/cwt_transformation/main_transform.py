# src/cwt_transformation/main_transform.py (最终可配置版)

import os
import sys
import numpy as np
import h5py
import pywt
from tqdm import tqdm

# 添加根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 不再直接从config导入常量，它们将通过config字典传入

def _calculate_scales(config):
    """
    根据配置计算用于CWT的尺度数组以匹配目标频率范围。
    
    Args:
        config (dict): 从 get_config() 生成的配置字典。
        
    Returns:
        np.ndarray: 用于pywt.cwt的尺度数组。
    """
    params = config['physical']
    cwt_freq_range_khz = params['cwt_freq_range_khz']
    cwt_wavelet_name = params['cwt_wavelet_name']
    sampling_rate_hz = params['sampling_rate_hz']
    cwt_scales_num = params['cwt_scales_num']
    
    f_min, f_max = cwt_freq_range_khz[0] * 1000, cwt_freq_range_khz[1] * 1000
    
    # 小波的中心频率
    central_freq = pywt.central_frequency(cwt_wavelet_name, precision=8)
    
    # 计算对应频率范围的尺度
    scale_max = central_freq * sampling_rate_hz / f_min
    scale_min = central_freq * sampling_rate_hz / f_max
    
    # 生成对数间隔的尺度数组
    scales = np.geomspace(scale_min, scale_max, cwt_scales_num)
    return scales

def run_cwt_transformation(config):
    """
    执行阶段二：读取对齐的波形数据，进行CWT变换，并保存尺度图。
    
    Args:
        config (dict): 从 get_config() 函数生成的配置字典。
    """
    paths = config['paths']
    params = config['physical']
    
    aligned_data_path = paths['aligned_data']
    scalogram_h5_path = paths['scalogram_h5']
    cwt_batch_size = params['cwt_batch_size']
    cwt_wavelet_name = params['cwt_wavelet_name']
    sampling_rate_hz = params['sampling_rate_hz']

    if not os.path.exists(aligned_data_path):
        print(f"错误: 对齐数据文件未找到于 {aligned_data_path}")
        print("请先为当前阵列运行 'preprocess' 步骤。")
        return

    if os.path.exists(scalogram_h5_path):
        print(f"CWT 结果: 尺度图文件已存在于 {scalogram_h5_path}. 跳过。")
        return

    print(f"为阵列 '{config['array_id']}' 加载对齐的波形数据...")
    with h5py.File(aligned_data_path, 'r') as hf:
        waveforms = hf['waveforms'][:]
    
    num_samples, num_timesteps = waveforms.shape
    scales = _calculate_scales(config)
    num_scales = len(scales)

    print(f"总计需要处理的样本数: {num_samples}")
    print(f"小波: {cwt_wavelet_name}, 尺度数: {num_scales}, 时间步: {num_timesteps}")

    # 创建HDF5文件以分批写入结果
    with h5py.File(scalogram_h5_path, 'w') as hf:
        scalogram_dset = hf.create_dataset(
            'scalograms',
            shape=(num_samples, num_scales, num_timesteps),
            dtype=np.float32,
            chunks=(1, num_scales, num_timesteps) # 优化读取性能
        )

        # 分批处理以防止内存溢出
        for i in tqdm(range(0, num_samples, cwt_batch_size), desc=f"为阵列_{config['array_id']}执行CWT"):
            batch_waveforms = waveforms[i:i + cwt_batch_size]
            
            # pywt.cwt 返回 (coeffs, freqs)，我们只需要系数
            coeffs, _ = pywt.cwt(batch_waveforms, scales, cwt_wavelet_name,
                                 sampling_period=1.0/sampling_rate_hz)
            
            # 计算幅值并存入HDF5文件
            # coeffs的形状是 (num_scales, num_samples_in_batch, num_timesteps)
            # 需要调整轴的顺序为 (num_samples_in_batch, num_scales, num_timesteps)
            scalograms_batch = np.abs(np.moveaxis(coeffs, 0, 1))
            scalogram_dset[i:i + cwt_batch_size, :, :] = scalograms_batch.astype(np.float32)

    print(f"已成功将尺度图保存至 {scalogram_h5_path}")


# --- 用于直接运行此脚本进行调试 (可选) ---
if __name__ == '__main__':
    # 这是一个示例，展示如何独立运行此脚本
    # 正常情况下，此脚本应由 main.py 调用
    print("【调试模式】")
    # 动态导入 get_config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import get_config
    
    # 选择一个阵列进行测试
    test_array_id = '03' 
    print(f"为阵列 {test_array_id} 执行CWT变换...")
    
    # 获取该阵列的配置
    debug_config = get_config(test_array_id)
    
    # 运行转换
    run_cwt_transformation(debug_config)