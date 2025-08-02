import h5py
import pandas as pd
import numpy as np

def save_data_to_h5(data_dict, file_path):
    """
    将包含Numpy数组或Pandas DataFrame的字典保存到HDF5文件。
    
    Args:
        data_dict (dict): 要保存的数据字典。
        file_path (str): HDF5文件路径。
    """
    with h5py.File(file_path, 'w') as hf:
        for key, value in data_dict.items():
            if isinstance(value, pd.DataFrame):
                hf.create_dataset(key, data=value.to_numpy())
                hf[key].attrs['columns'] = list(value.columns)
                hf[key].attrs['type'] = 'dataframe'
            elif isinstance(value, np.ndarray):
                hf.create_dataset(key, data=value)
                hf[key].attrs['type'] = 'ndarray'
            else:
                print(f"Warning: Skipping unsupported type {type(value)} for key '{key}'")


def load_data_from_h5(file_path):
    """
    从HDF5文件加载数据到字典。
    
    Args:
        file_path (str): HDF5文件路径。
        
    Returns:
        dict: 加载后的数据字典。
    """
    data_dict = {}
    with h5py.File(file_path, 'r') as hf:
        for key in hf.keys():
            if 'type' in hf[key].attrs and hf[key].attrs['type'] == 'dataframe':
                data = hf[key][:]
                columns = hf[key].attrs['columns']
                data_dict[key] = pd.DataFrame(data, columns=columns)
            else:
                data_dict[key] = hf[key][:]
    return data_dict