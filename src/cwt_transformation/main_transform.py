import os
import sys
import numpy as np
import pywt
import h5py
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (ALIGNED_DATA_PATH, SCALOGRAM_DATA_PATH, WAVELET_NAME,
                    SAMPLING_RATE_HZ, TARGET_FREQ_RANGE_KHZ, NUM_SCALES)
from utils.file_io import load_data_from_h5

def _calculate_cwt_scales():
    """
    Calculates the appropriate scales for CWT based on the target frequency range.
    (Corrected version using the direct physical formula)
    """
    print("Calculating CWT scales for target frequency range...")
    
    target_freq_min = TARGET_FREQ_RANGE_KHZ[0] * 1000
    target_freq_max = TARGET_FREQ_RANGE_KHZ[1] * 1000
    
    # Create a logarithmic space of desired frequencies
    frequencies_hz = np.logspace(np.log10(target_freq_min), np.log10(target_freq_max), NUM_SCALES)
    
    # Get the central frequency of the mother wavelet
    # This is a characteristic property of the chosen wavelet
    central_frequency = pywt.central_frequency(WAVELET_NAME)
    
    # The physical formula relating scale, frequency, and sampling rate
    scales = central_frequency * SAMPLING_RATE_HZ / frequencies_hz
    
    # For consistency and easier plotting, sort scales in ascending order
    # and reorder frequencies to match.
    sort_indices = np.argsort(scales)
    scales = scales[sort_indices]
    frequencies_hz = frequencies_hz[sort_indices]
    
    print(f"  - Wavelet: {WAVELET_NAME} (Central Freq: {central_frequency:.4f})")
    print(f"  - Calculated {len(scales)} scales (from {scales.min():.2f} to {scales.max():.2f}) for frequency range {TARGET_FREQ_RANGE_KHZ} kHz.")
    
    return scales, frequencies_hz

def run_cwt_transformation():
    """
    Executes the wavelet transform to convert waveforms into scalograms.
    """
    if os.path.exists(SCALOGRAM_DATA_PATH):
        print(f"Scalogram data file already exists at {SCALOGRAM_DATA_PATH}. Skipping.")
        return

    print("\n--- Starting CWT Transformation ---")
    
    # 1. Load preprocessed data
    print("Step 1: Loading aligned data...")
    aligned_data = load_data_from_h5(ALIGNED_DATA_PATH)
    waveforms = aligned_data['waveforms']
    csi_labels = aligned_data['csi_labels']
    
    num_samples, num_timesteps = waveforms.shape
    print(f"  - Loaded {num_samples} waveforms.")
    
    # 2. Calculate CWT scales
    scales, frequencies_hz = _calculate_cwt_scales()
    num_scales = len(scales)
    
    # 3. Initialize the output HDF5 file for incremental writing
    with h5py.File(SCALOGRAM_DATA_PATH, 'w') as hf:
        scalogram_dset = hf.create_dataset(
            'scalograms',
            shape=(num_samples, num_scales, num_timesteps),
            maxshape=(None, num_scales, num_timesteps),
            dtype='float32',
            chunks=(1, num_scales, num_timesteps)
        )
        hf.create_dataset('csi_labels', data=csi_labels)
        hf.create_dataset('frequencies_hz', data=frequencies_hz)
        if 'metadata' in aligned_data:
             hf.create_dataset('metadata', data=aligned_data['metadata'])
             
        # 4. Perform CWT in a memory-efficient loop
        print("Step 2: Performing CWT and saving scalograms...")
        
        for i in tqdm(range(num_samples), desc="Generating Scalograms"):
            waveform = waveforms[i, :]
            
            # Perform CWT
            coeffs, _ = pywt.cwt(waveform, scales, WAVELET_NAME)
            
            # Calculate magnitude to get the scalogram
            scalogram = np.abs(coeffs).astype('float32')
            
            # Write the single scalogram to the HDF5 dataset
            scalogram_dset[i, :, :] = scalogram

    print(f"\nSuccessfully generated and saved scalogram dataset to {SCALOGRAM_DATA_PATH}")

if __name__ == '__main__':
    run_cwt_transformation()