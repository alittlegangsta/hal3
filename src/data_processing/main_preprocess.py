import os
import sys
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d # 导入新的插值库
from tqdm import tqdm

# 将src目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (ULTRASONIC_PATH, SONIC_PATH, ORIENTATION_PATH, ALIGNED_DATA_PATH,
                    DEPTH_RANGE, SAMPLING_RATE_HZ, FILTER_CUTOFF_HZ, FILTER_ORDER,
                    DEPTH_RESOLUTION_FT, ZC_THRESHOLD, VERTICAL_WINDOW_SIZE,
                    BALANCE_CSI_THRESHOLD, MAX_LOW_CSI_SAMPLES, PLOT_DIR)
from utils.file_io import save_data_to_h5
from utils.plotting import plot_csi_distribution

def _load_and_validate_raw_data():
    """加载并验证所有原始.mat文件。"""
    print("Step 1: Loading and validating raw data...")
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
    print("Step 2: Applying high-pass filter to sonic waveforms...")
    b, a = butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype='high', fs=SAMPLING_RATE_HZ)
    wave_keys = [key for key in sonic_data.keys() if key.startswith('WaveRng')]
    for key in tqdm(wave_keys, desc="Filtering waveforms"):
        sonic_data[key] = filtfilt(b, a, sonic_data[key], axis=0)
    return sonic_data

def _align_data_to_uniform_depth(ultrasonic, sonic, orientation):
    """将所有数据插值到统一的深度轴。"""
    print("Step 3: Aligning all data to a uniform depth axis...")
    uniform_depth = np.arange(DEPTH_RANGE[0], DEPTH_RANGE[1], DEPTH_RESOLUTION_FT)
    
    # 检查原始深度轴是否单调递增，这是插值的要求
    zc_original_depth = ultrasonic['Depth']
    if not np.all(np.diff(zc_original_depth) >= 0):
        print("  - Sorting ultrasonic data by depth as it is not monotonic.")
        sort_indices = np.argsort(zc_original_depth)
        zc_original_depth = zc_original_depth[sort_indices]
        ultrasonic['Zc'] = ultrasonic['Zc'][:, sort_indices]

    sonic_original_depth = sonic['Depth']
    if not np.all(np.diff(sonic_original_depth) >= 0):
        print("  - Sorting sonic data by depth as it is not monotonic.")
        sort_indices = np.argsort(sonic_original_depth)
        sonic_original_depth = sonic_original_depth[sort_indices]
        for key in [k for k in sonic.keys() if k.startswith('WaveRng')]:
            sonic[key] = sonic[key][:, sort_indices]

    # 插值方位数据
    aligned_orientation = pd.DataFrame({
        'Inc': np.interp(uniform_depth, orientation['Depth_inc'], orientation['Inc']),
        'RelBearing': np.interp(uniform_depth, orientation['Depth_inc'], orientation['RelBearing'])
    })
    
    # 使用 Scipy.interp1d 进行 Zc 数据插值
    zc_original_data = ultrasonic['Zc']
    aligned_zc = np.zeros((zc_original_data.shape[0], len(uniform_depth)))
    for i in tqdm(range(zc_original_data.shape[0]), desc="Interpolating Zc data (SciPy)"):
        f_zc = interp1d(zc_original_depth, zc_original_data[i, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        aligned_zc[i, :] = f_zc(uniform_depth)

    # 使用 Scipy.interp1d 进行声波数据插值
    aligned_sonic = {}
    wave_keys = [key for key in sonic.keys() if key.startswith('WaveRng')]
    for key in tqdm(wave_keys, desc="Interpolating sonic data (SciPy)"):
        wave_data = sonic[key]
        interp_wave = np.zeros((wave_data.shape[0], len(uniform_depth)))
        for j in range(wave_data.shape[0]):
            f_wave = interp1d(sonic_original_depth, wave_data[j, :], kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_wave[j, :] = f_wave(uniform_depth)
        aligned_sonic[key] = interp_wave
        
    return uniform_depth, aligned_zc, aligned_sonic, aligned_orientation

def _calculate_csi_and_create_dataset(uniform_depth, zc, sonic, orientation):
    """执行方位校正，匹配扇区，并计算CSI。"""
    print("Step 4: Performing azimuthal correction and calculating CSI...")
    # ... (此函数的其余部分代码保持完全不变) ...
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

    return np.array(waveforms), np.array(csi_labels), np.array(metadata)


def _balance_dataset(waveforms, csi_labels, metadata):
    """根据CSI值平衡数据集。"""
    if MAX_LOW_CSI_SAMPLES is None:
        return waveforms, csi_labels, metadata
    print(f"Step 5: Balancing dataset...")
    full_dataset = pd.DataFrame(metadata, columns=['depth', 'receiver_idx'])
    full_dataset['csi'] = csi_labels
    low_csi_df = full_dataset[full_dataset['csi'] < BALANCE_CSI_THRESHOLD]
    high_csi_df = full_dataset[full_dataset['csi'] >= BALANCE_CSI_THRESHOLD]
    print(f"  - Original counts: Low CSI (<{BALANCE_CSI_THRESHOLD}): {len(low_csi_df)}, High CSI: {len(high_csi_df)}")
    if len(low_csi_df) > MAX_LOW_CSI_SAMPLES:
        low_csi_df = low_csi_df.sample(n=MAX_LOW_CSI_SAMPLES, random_state=42)
        print(f"  - Downsampling Low CSI samples to {MAX_LOW_CSI_SAMPLES}")
    balanced_df = pd.concat([low_csi_df, high_csi_df]).sort_index()
    balanced_indices = balanced_df.index.values
    final_waveforms = waveforms[balanced_indices]
    final_csi = csi_labels[balanced_indices]
    final_metadata = metadata[balanced_indices]
    print(f"  - Final counts: Low CSI: {len(low_csi_df)}, High CSI: {len(high_csi_df)}, Total: {len(final_csi)}")
    return final_waveforms, final_csi, final_metadata

def run_preprocessing():
    """执行完整的预处理流程。"""
    if os.path.exists(ALIGNED_DATA_PATH):
        print(f"Processed data file already exists at {ALIGNED_DATA_PATH}. Skipping.")
        return
    ultrasonic, sonic, orientation = _load_and_validate_raw_data()
    sonic_filtered = _apply_high_pass_filter(sonic)
    uniform_depth, aligned_zc, aligned_sonic, aligned_orientation = \
        _align_data_to_uniform_depth(ultrasonic, sonic_filtered, orientation)
    waveforms, csi_labels, metadata = _calculate_csi_and_create_dataset(
        uniform_depth, aligned_zc, aligned_sonic, aligned_orientation
    )
    waveforms, csi_labels, metadata = _balance_dataset(waveforms, csi_labels, metadata)
    print("Step 6: Saving processed and aligned data...")
    data_to_save = {
        'waveforms': waveforms,
        'csi_labels': csi_labels,
        'metadata': metadata
    }
    save_data_to_h5(data_to_save, ALIGNED_DATA_PATH)
    print(f"Successfully saved aligned data to {ALIGNED_DATA_PATH}")
    plot_csi_distribution(csi_labels, save_path=os.path.join(PLOT_DIR, 'csi_distribution.png'))

if __name__ == '__main__':
    run_preprocessing()