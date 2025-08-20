# src/data_processing/main_preprocess.py (最终物理精确版)

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ... (这个文件的其他函数保持不变，只修改 _load_and_validate_raw_data)

def _load_and_validate_raw_data(config):
    """加载、验证、应用深度校正并裁剪所有原始.mat文件。"""
    print("Step 1/4: Loading, validating, and cropping raw data...")
    paths = config['paths']
    timesteps = config['physical']['waveform_timesteps']
    array_id = int(config['array_id'])
    
    # 加载超声数据
    cast_mat = scipy.io.loadmat(paths['ultrasonic'], squeeze_me=True)
    cast_struct_obj = cast_mat['CAST']
    cast_fields = cast_struct_obj.dtype.names
    cast_struct_tuple = cast_struct_obj.item()
    depth_idx = cast_fields.index('Depth')
    zc_idx = cast_fields.index('Zc')
    ultrasonic_data = {
        'Depth': cast_struct_tuple[depth_idx].flatten(),
        'Zc': cast_struct_tuple[zc_idx]
    }
    
    # 加载声波数据
    sonic_mat = scipy.io.loadmat(paths['sonic'], squeeze_me=True)
    sonic_struct_key = config['sonic_struct_key']
    sonic_struct_obj = sonic_mat[sonic_struct_key]
    sonic_struct_tuple = sonic_struct_obj.item()
    sonic_fields = sonic_struct_obj.dtype.names
    
    sonic_data = {'Depth': sonic_struct_tuple[sonic_fields.index('Depth')].flatten()}
    
    # --- 核心修改：应用物理深度校正 ---
    depth_offset = (7 - array_id) * 0.5
    if depth_offset != 0:
        sonic_data['Depth'] += depth_offset
        print(f"  - Applied a physical depth offset of {depth_offset:.2f} ft for array {config['array_id']}.")
    else:
        print(f"  - Array {config['array_id']} is the reference, no depth offset applied.")
    # --- 修正结束 ---
        
    wave_keys = [key for key in sonic_fields if key.startswith('WaveRng')]
    for key in wave_keys:
        full_waveform = sonic_struct_tuple[sonic_fields.index(key)]
        sonic_data[key] = full_waveform[:timesteps, :]
    
    print(f"  - Sonic waveforms have been cropped to the first {timesteps} timesteps (4ms).")

    # 加载方位数据
    orientation_mat = scipy.io.loadmat(paths['orientation'])
    orientation_data = {
        'Depth_inc': orientation_mat['Depth_inc'].flatten(),
        'Inc': orientation_mat['Inc'].flatten(),
        'RelBearing': orientation_mat['RelBearing'].flatten()
    }
    return ultrasonic_data, sonic_data, orientation_data

# ... (文件其余部分与之前修正版一致)
# --- 为了完整性，粘贴整个文件的其余部分 ---

def _apply_high_pass_filter(sonic_data, config):
    print("Step 2/4: Applying high-pass filter to sonic waveforms...")
    params = config['physical']
    b, a = butter(params['filter_order'], params['filter_cutoff_hz'], btype='high', fs=params['sampling_rate_hz'])
    wave_keys = [key for key in sonic_data.keys() if key.startswith('WaveRng')]
    for key in tqdm(wave_keys, desc="Filtering waveforms"):
        sonic_data[key] = filtfilt(b, a, sonic_data[key], axis=0)
    return sonic_data

def _align_data_to_uniform_depth(ultrasonic, sonic, orientation, config):
    print("Step 3/4: Aligning all data to a uniform depth axis...")
    params = config['physical']
    depth_range = params['depth_range']
    depth_resolution = params['depth_resolution_ft']
    uniform_depth = np.arange(depth_range[0], depth_range[1], depth_resolution)
    
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

def _calculate_csi_and_create_dataset(uniform_depth, zc, sonic, orientation, config):
    print("Step 4/4: Performing azimuthal correction and calculating CSI...")
    params = config['physical']
    zc_threshold = params['zc_threshold']
    vertical_window_size = params['vertical_window_size']
    
    num_depths = len(uniform_depth)
    num_receivers = 8
    receiver_angles = np.arange(num_receivers) * 45
    
    instrument_ref_azimuth = orientation['RelBearing'].values
    receiver_abs_azimuths = np.zeros((num_depths, num_receivers))
    for i in range(num_receivers):
        receiver_abs_azimuths[:, i] = (instrument_ref_azimuth + receiver_angles[i]) % 360

    zc_binary = (zc < zc_threshold).astype(np.int8)
    
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

            depth_min = depth_val - vertical_window_size / 2
            depth_max = depth_val + vertical_window_size / 2
            
            depth_indices = np.where((uniform_depth >= depth_min) & (uniform_depth <= depth_max))[0]
            
            if azimuth_min < azimuth_max:
                azimuth_indices = np.where((zc_azimuth_axis >= azimuth_min) & (zc_azimuth_axis <= azimuth_max))[0]
            else: 
                azimuth_indices = np.where((zc_azimuth_axis >= azimuth_min) | (zc_azimuth_axis <= azimuth_max))[0]

            if len(depth_indices) == 0 or len(azimuth_indices) == 0:
                csi = 0.0
            else:
                relevant_zc_binary_region = zc_binary[azimuth_indices, :][:, depth_indices]
                csi = np.mean(relevant_zc_binary_region)

            waveforms.append(waveform)
            csi_labels.append(csi)
            metadata.append([depth_val, rec_idx])

    return np.array(waveforms, dtype=np.float32), np.array(csi_labels, dtype=np.float32), np.array(metadata)


def run_stage1_preprocessing(config):
    aligned_data_path = config['paths']['aligned_data']
    if os.path.exists(aligned_data_path):
        print(f"Stage 1 Result: Processed data file already exists at {aligned_data_path}. Skipping.")
        return
        
    ultrasonic, sonic, orientation = _load_and_validate_raw_data(config)
    sonic_filtered = _apply_high_pass_filter(sonic, config)
    uniform_depth, aligned_zc, aligned_sonic, aligned_orientation = \
        _align_data_to_uniform_depth(ultrasonic, sonic_filtered, orientation, config)
    
    waveforms, csi_labels, metadata = _calculate_csi_and_create_dataset(
        uniform_depth, aligned_zc, aligned_sonic, aligned_orientation, config
    )
    
    print("Saving processed and aligned data...")
    with h5py.File(aligned_data_path, 'w') as hf:
        hf.create_dataset('waveforms', data=waveforms)
        hf.create_dataset('csi_labels', data=csi_labels)
        hf.create_dataset('metadata', data=metadata)
        
    print(f"Successfully saved aligned data to {aligned_data_path}")

def run_stage2_split_and_normalize(config):
    paths = config['paths']
    modeling_params = config['modeling']
    data_proc_params = config['data_processing']
    
    print("Loading CSI labels for splitting...")
    try:
        with h5py.File(paths['aligned_data'], 'r') as hf:
            all_csi = hf['csi_labels'][:]
    except (FileNotFoundError, KeyError):
        print(f"Error: Could not load 'csi_labels' from {paths['aligned_data']}.")
        return

    indices = np.arange(len(all_csi))
    csi_bins_def = data_proc_params['csi_bins']
    csi_labels_binned = np.digitize(all_csi, bins=[c[2] for c in csi_bins_def[:-1]])
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=modeling_params['validation_split'],
        stratify=csi_labels_binned,
        random_state=modeling_params['random_seed']
    )

    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")

    print("Calculating normalization stats (mean, std) from training set scalograms...")
    try:
        with h5py.File(paths['scalogram_h5'], 'r') as hf:
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
        print(f"Error: Could not load 'scalograms' from {paths['scalogram_h5']}.")
        print("Please ensure you have run the CWT transformation script to generate this file.")
        return

    print(f"Calculation complete: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    with open(paths['norm_stats'], 'w') as f:
        json.dump({'mean': mean_val, 'std': std_val}, f)
    print(f"Normalization stats saved to: {paths['norm_stats']}")

    np.savez(paths['split_indices'], train_indices=train_indices, val_indices=val_indices)
    print(f"Dataset split indices saved to: {paths['split_indices']}")

def run_stage3_normalize_data(config):
    paths = config['paths']
    training_ready_path = paths['training_ready_data']
    
    if os.path.exists(training_ready_path):
        print(f"Final data file already exists at {training_ready_path}. Skipping.")
        return

    print("Verifying required files exist...")
    required_files = [paths['scalogram_h5'], paths['norm_stats'], paths['split_indices']]
    for f_path in required_files:
        if not os.path.exists(f_path):
            print(f"Error: Required file not found at {f_path}")
            return

    print("Loading split indices and normalization stats...")
    indices_data = np.load(paths['split_indices'])
    train_indices = indices_data['train_indices']
    val_indices = indices_data['val_indices']
    
    with open(paths['norm_stats'], 'r') as f:
        norm_stats = json.load(f)
    mean = norm_stats['mean']
    std = norm_stats['std']

    batch_size = 1024

    print(f"Opening data files and preparing for chunked processing (batch size: {batch_size})...")
    with h5py.File(paths['scalogram_h5'], 'r') as hf_in, \
         h5py.File(paths['aligned_data'], 'r') as hf_labels, \
         h5py.File(training_ready_path, 'w') as hf_out:

        x_train_shape = (len(train_indices), hf_in['scalograms'].shape[1], hf_in['scalograms'].shape[2])
        x_val_shape = (len(val_indices), hf_in['scalograms'].shape[1], hf_in['scalograms'].shape[2])

        dset_x_train = hf_out.create_dataset('x_train', shape=x_train_shape, dtype=np.float32, chunks=True)
        dset_y_train = hf_out.create_dataset('y_train', shape=(len(train_indices),), dtype=np.float32)
        dset_x_val = hf_out.create_dataset('x_val', shape=x_val_shape, dtype=np.float32, chunks=True)
        dset_y_val = hf_out.create_dataset('y_val', shape=(len(val_indices),), dtype=np.float32)

        print(f"Processing {len(train_indices)} training samples in chunks...")
        sorted_train_indices = np.sort(train_indices)
        for i in tqdm(range(0, len(sorted_train_indices), batch_size), desc="Normalizing Train Set"):
            batch_indices = sorted_train_indices[i:i + batch_size]
            
            x_batch = hf_in['scalograms'][batch_indices, :, :]
            y_batch = hf_labels['csi_labels'][batch_indices]
            
            x_batch_norm = (x_batch - mean) / std
            
            start_index = i
            end_index = i + len(batch_indices)
            dset_x_train[start_index:end_index, :, :] = x_batch_norm
            dset_y_train[start_index:end_index] = y_batch
        
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