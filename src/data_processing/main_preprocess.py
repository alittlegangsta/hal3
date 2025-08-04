# src/data_processing/main_preprocess.py

import os
import sys
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
import json

# 添加根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 从config导入所有需要的变量
from config import (
    ULTRASONIC_PATH, SONIC_PATH, ORIENTATION_PATH, ALIGNED_DATA_PATH,
    DEPTH_RANGE, SAMPLING_RATE_HZ, FILTER_CUTOFF_HZ, FILTER_ORDER,
    DEPTH_RESOLUTION_FT, ZC_THRESHOLD, VERTICAL_WINDOW_SIZE,
    PLOT_DIR, SCALOGRAM_H5_PATH, NORM_STATS_PATH, SPLIT_INDICES_PATH,
    # --- 关键修正：在这里添加缺失的变量 ---
    TRAINING_READY_DATA_PATH,
    # -----------------------------------------
    MODELING_CONFIG, DATA_PROCESSING_CONFIG
)
# 假设您有这些工具函数
# from utils.file_io import save_data_to_h5
# from utils.plotting import plot_csi_distribution

# ==============================================================================
# 阶段一：函数 (从原始数据到对齐的波形)
# ==============================================================================

def _load_and_validate_raw_data():
    """加载并验证所有原始.mat文件。"""
    print("Step 1/4: Loading and validating raw data...")
    cast_mat = scipy.io.loadmat(ULTRASONIC_PATH, squeeze_me=True)
    cast_struct_obj = cast_mat['CAST']
    cast_fields = cast_struct_obj.dtype.names
    cast_struct_tuple = cast_struct_obj.item()
    depth_idx_cast = cast_fields.index('Depth')
    zc_idx = cast_fields.index('Zc')
    ultrasonic_data = {
        'Depth': cast_struct_tuple[depth_idx_cast].flatten(),
        'Zc': cast_struct_tuple[zc_idx]
    }
    sonic_mat = scipy.io.loadmat(SONIC_PATH, squeeze_me=True)
    sonic_struct_obj = sonic_mat['XSILMR03']
    sonic_fields = sonic_struct_obj.dtype.names
    sonic_struct_tuple = sonic_struct_obj.item()
    depth_idx_sonic = sonic_fields.index('Depth')
    sonic_data = {'Depth': sonic_struct_tuple[depth_idx_sonic].flatten()}
    wave_keys = [key for key in sonic_fields if key.startswith('WaveRng')]
    for key in wave_keys:
        key_idx = sonic_fields.index(key)
        sonic_data[key] = sonic_struct_tuple[key_idx]
    orientation_mat = scipy.io.loadmat(ORIENTATION_PATH)
    orientation_data = {
        'Depth_inc': orientation_mat['Depth_inc'].flatten(),
        'Inc': orientation_mat['Inc'].flatten(),
        'RelBearing': orientation_mat['RelBearing'].flatten()
    }
    return ultrasonic_data, sonic_data, orientation_data

def _apply_high_pass_filter(sonic_data):
    """对所有声波波形应用高通滤波器。"""
    print("Step 2/4: Applying high-pass filter to sonic waveforms...")
    b, a = butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype='high', fs=SAMPLING_RATE_HZ)
    wave_keys = [key for key in sonic_data.keys() if key.startswith('WaveRng')]
    for key in tqdm(wave_keys, desc="Filtering waveforms"):
        sonic_data[key] = filtfilt(b, a, sonic_data[key], axis=0)
    return sonic_data

def _align_data_to_uniform_depth(ultrasonic, sonic, orientation):
    """将所有数据插值到统一的深度轴。"""
    print("Step 3/4: Aligning all data to a uniform depth axis...")
    uniform_depth = np.arange(DEPTH_RANGE[0], DEPTH_RANGE[1], DEPTH_RESOLUTION_FT)
    
    # 保证深度单调性
    for name, data_dict, depth_key in [('ultrasonic', ultrasonic, 'Depth'), ('sonic', sonic, 'Depth')]:
        original_depth = data_dict[depth_key]
        if not np.all(np.diff(original_depth) >= 0):
            print(f"  - Sorting {name} data by depth as it is not monotonic.")
            sort_indices = np.argsort(original_depth)
            data_dict[depth_key] = original_depth[sort_indices]
            for key, value in data_dict.items():
                if key != depth_key and isinstance(value, np.ndarray) and value.ndim == 2:
                     data_dict[key] = value[:, sort_indices]

    aligned_orientation = pd.DataFrame({
        'Inc': np.interp(uniform_depth, orientation['Depth_inc'], orientation['Inc']),
        'RelBearing': np.interp(uniform_depth, orientation['Depth_inc'], orientation['RelBearing'])
    })
    
    f_zc = interp1d(ultrasonic['Depth'], ultrasonic['Zc'], kind='linear', bounds_error=False, fill_value='extrapolate', axis=1)
    aligned_zc = f_zc(uniform_depth)

    aligned_sonic = {}
    wave_keys = [key for key in sonic.keys() if key.startswith('WaveRng')]
    for key in tqdm(wave_keys, desc="Interpolating sonic data"):
        f_wave = interp1d(sonic['Depth'], sonic[key], kind='linear', bounds_error=False, fill_value='extrapolate', axis=1)
        aligned_sonic[key] = f_wave(uniform_depth)
        
    return uniform_depth, aligned_zc, aligned_sonic, aligned_orientation

def _calculate_csi_and_create_dataset(uniform_depth, zc, sonic, orientation):
    """执行方位校正，匹配扇区，并计算CSI。"""
    print("Step 4/4: Performing azimuthal correction and calculating CSI...")
    num_depths = len(uniform_depth)
    num_receivers = 8
    receiver_angles = np.arange(num_receivers) * 45
    
    instrument_ref_azimuth = orientation['RelBearing'].values
    receiver_abs_azimuths = np.zeros((num_depths, num_receivers))
    for i in range(num_receivers):
        receiver_abs_azimuths[:, i] = (instrument_ref_azimuth + receiver_angles[i]) % 360

    zc_binary = (zc < ZC_THRESHOLD).astype(np.int8)
    
    waveforms = []
    csi_labels = []
    metadata = []

    zc_azimuth_axis = np.arange(0, 360, 2)
    wave_keys = sorted([key for key in sonic.keys() if key.startswith('WaveRng')])

    for depth_idx in tqdm(range(num_depths), desc="Calculating CSI for each waveform"):
        depth_val = uniform_depth[depth_idx]
        for rec_idx in range(num_receivers):
            waveform = sonic[wave_keys[rec_idx]][:, depth_idx]
            
            center_azimuth = receiver_abs_azimuths[depth_idx, rec_idx]
            azimuth_min = (center_azimuth - 22.5) % 360
            azimuth_max = (center_azimuth + 22.5) % 360

            depth_min = depth_val - VERTICAL_WINDOW_SIZE / 2
            depth_max = depth_val + VERTICAL_WINDOW_SIZE / 2
            
            depth_indices = np.where((uniform_depth >= depth_min) & (uniform_depth <= depth_max))[0]
            
            if azimuth_min < azimuth_max:
                azimuth_indices = np.where((zc_azimuth_axis >= azimuth_min) & (zc_azimuth_axis <= azimuth_max))[0]
            else: 
                azimuth_indices = np.where((zc_azimuth_axis >= azimuth_min) | (zc_azimuth_axis <= azimuth_max))[0]

            if len(depth_indices) == 0 or len(azimuth_indices) == 0:
                csi = 0.0
            else:
                relevant_zc_binary_region = zc_binary[np.ix_(azimuth_indices, depth_indices)]
                csi = np.mean(relevant_zc_binary_region)

            waveforms.append(waveform)
            csi_labels.append(csi)
            metadata.append([depth_val, rec_idx])

    return np.array(waveforms, dtype=np.float32), np.array(csi_labels, dtype=np.float32), np.array(metadata)


def run_stage1_preprocessing():
    """执行完整的预处理流程 (阶段一)。"""
    if os.path.exists(ALIGNED_DATA_PATH):
        print(f"Stage 1 Result: Processed data file already exists at {ALIGNED_DATA_PATH}. Skipping.")
        return
        
    ultrasonic, sonic, orientation = _load_and_validate_raw_data()
    sonic_filtered = _apply_high_pass_filter(sonic)
    uniform_depth, aligned_zc, aligned_sonic, aligned_orientation = \
        _align_data_to_uniform_depth(ultrasonic, sonic_filtered, orientation)
    
    waveforms, csi_labels, metadata = _calculate_csi_and_create_dataset(
        uniform_depth, aligned_zc, aligned_sonic, aligned_orientation
    )
    
    print("Saving processed and aligned data...")
    with h5py.File(ALIGNED_DATA_PATH, 'w') as hf:
        hf.create_dataset('waveforms', data=waveforms)
        hf.create_dataset('csi_labels', data=csi_labels)
        hf.create_dataset('metadata', data=metadata)
        
    print(f"Successfully saved aligned data to {ALIGNED_DATA_PATH}")

# ==============================================================================
# 阶段三：函数 (数据集划分与归一化)
# ==============================================================================

def run_stage2_split_and_normalize():
    """
    (阶段三) 加载CWT后的尺度图数据，执行分层抽样来划分训练/验证集，
    并仅使用训练集数据计算和保存归一化统计数据。
    """
    print("Loading CSI labels for splitting...")
    try:
        with h5py.File(ALIGNED_DATA_PATH, 'r') as hf:
            all_csi = hf['csi_labels'][:]
    except (FileNotFoundError, KeyError):
        print(f"Error: Could not load 'csi_labels' from {ALIGNED_DATA_PATH}.")
        print("Please ensure you have successfully run Stage 1 ('preprocess') first.")
        return

    num_samples = len(all_csi)
    indices = np.arange(num_samples)

    csi_bins_def = DATA_PROCESSING_CONFIG['csi_bins']
    # 使用元组的上界进行分箱
    csi_labels_binned = np.digitize(all_csi, bins=[c[2] for c in csi_bins_def[:-1]])
    
    print("Performing stratified split based on CSI bins...")
    train_indices, val_indices = train_test_split(
        indices,
        test_size=MODELING_CONFIG['validation_split'],
        stratify=csi_labels_binned,
        random_state=MODELING_CONFIG['random_seed']
    )

    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    print("Calculating normalization stats (mean, std) from training set scalograms...")
    try:
        with h5py.File(SCALOGRAM_H5_PATH, 'r') as hf:
            train_scalograms_dset = hf['scalograms']
            
            batch_size = 512
            sum_val = 0.0
            sum_sq_val = 0.0
            total_pixels = 0
            num_train_samples = len(train_indices)
            
            for i in tqdm(range(0, num_train_samples, batch_size), desc="Calculating Mean"):
                batch_indices = np.sort(train_indices[i:i+batch_size])
                batch_data = train_scalograms_dset[batch_indices, :, :]
                sum_val += np.sum(batch_data, dtype=np.float64)
                total_pixels += batch_data.size
            mean_val = sum_val / total_pixels

            for i in tqdm(range(0, num_train_samples, batch_size), desc="Calculating Std Dev"):
                batch_indices = np.sort(train_indices[i:i+batch_size])
                batch_data = train_scalograms_dset[batch_indices, :, :]
                sum_sq_val += np.sum(((batch_data - mean_val) ** 2), dtype=np.float64)
            std_val = np.sqrt(sum_sq_val / total_pixels)

    except (FileNotFoundError, KeyError):
        print(f"Error: Could not load 'scalograms' from {SCALOGRAM_H5_PATH}.")
        print("Please ensure you have run the CWT transformation script to generate this file.")
        return

    print(f"Calculation complete: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    with open(NORM_STATS_PATH, 'w') as f:
        json.dump({'mean': mean_val, 'std': std_val}, f)
    print(f"Normalization stats saved to: {NORM_STATS_PATH}")

    np.savez(SPLIT_INDICES_PATH, train_indices=train_indices, val_indices=val_indices)
    print(f"Dataset split indices saved to: {SPLIT_INDICES_PATH}")

# ==============================================================================
# 最终安全版：阶段四函数 (创建最终的训练就绪数据)
# ==============================================================================
def run_stage3_normalize_data():
    """
    (阶段四) 读取尺度图，【分块】应用标准化，并保存为最终的训练就绪HDF5文件。
    此版本经过内存优化，可避免内存爆炸。
    """
    print("Verifying required files exist...")
    required_files = [SCALOGRAM_H5_PATH, NORM_STATS_PATH, SPLIT_INDICES_PATH]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"Error: Required file not found at {f_path}")
            print("Please ensure you have run 'cwt' and 'split' steps first.")
            return
            
    if os.path.exists(TRAINING_READY_DATA_PATH):
        print(f"Final data file already exists at {TRAINING_READY_DATA_PATH}. Skipping.")
        return

    print("Loading split indices and normalization stats...")
    indices_data = np.load(SPLIT_INDICES_PATH)
    train_indices = indices_data['train_indices']
    val_indices = indices_data['val_indices']
    
    with open(NORM_STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
    mean = norm_stats['mean']
    std = norm_stats['std']

    # --- 定义处理参数 ---
    batch_size = 1024 # 定义一个合理的批处理大小，可以根据您的RAM调整

    print(f"Opening data files and preparing for chunked processing (batch size: {batch_size})...")
    with h5py.File(SCALOGRAM_H5_PATH, 'r') as hf_in, \
         h5py.File(ALIGNED_DATA_PATH, 'r') as hf_labels, \
         h5py.File(TRAINING_READY_DATA_PATH, 'w') as hf_out:

        # --- 创建目标数据集 ---
        # 预先定义好最终文件的结构和尺寸
        x_train_shape = (len(train_indices), hf_in['scalograms'].shape[1], hf_in['scalograms'].shape[2])
        x_val_shape = (len(val_indices), hf_in['scalograms'].shape[1], hf_in['scalograms'].shape[2])

        dset_x_train = hf_out.create_dataset('x_train', shape=x_train_shape, dtype=np.float32, chunks=True)
        dset_y_train = hf_out.create_dataset('y_train', shape=(len(train_indices),), dtype=np.float32)
        dset_x_val = hf_out.create_dataset('x_val', shape=x_val_shape, dtype=np.float32, chunks=True)
        dset_y_val = hf_out.create_dataset('y_val', shape=(len(val_indices),), dtype=np.float32)

        # --- 按块处理训练集 ---
        print(f"Processing {len(train_indices)} training samples in chunks...")
        sorted_train_indices = np.sort(train_indices) # 排序以优化HDF5读取
        for i in tqdm(range(0, len(sorted_train_indices), batch_size), desc="Normalizing Train Set"):
            batch_indices = sorted_train_indices[i:i + batch_size]
            
            # 1. 读取一小批数据
            x_batch = hf_in['scalograms'][batch_indices, :, :]
            y_batch = hf_labels['csi_labels'][batch_indices]
            
            # 2. 在内存中对这一小批数据进行标准化
            x_batch_norm = (x_batch - mean) / std
            
            # 3. 将处理好的小批数据写入到新文件的对应位置
            start_index = i
            end_index = i + len(batch_indices)
            dset_x_train[start_index:end_index, :, :] = x_batch_norm
            dset_y_train[start_index:end_index] = y_batch

        # --- 按块处理验证集 ---
        print(f"Processing {len(val_indices)} validation samples in chunks...")
        sorted_val_indices = np.sort(val_indices)
        for i in tqdm(range(0, len(sorted_val_indices), batch_size), desc="Normalizing Validation Set"):
            batch_indices = sorted_val_indices[i:i + batch_size]
            
            x_batch = hf_in['scalograms'][batch_indices, :, :]
            y_batch = hf_labels['csi_labels'][batch_indices]
            
            x_batch_norm = (x_batch - mean) / std
            
            start_index = i
            end_index = i + len(batch_indices)
            dset_x_val[start_index:end_index, :, :] = x_batch_norm
            dset_y_val[start_index:end_index] = y_batch

    print("Successfully created the final training-ready data file with memory optimization.")