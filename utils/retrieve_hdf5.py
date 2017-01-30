import numpy as np
import h5py
 
def extract_hdf5(file_path):
    with h5py.File(file_path,'r') as hdf5_data:
        """
        hdf5_data.keys() = (data, label)
        data_arr.squeeze().shape = (number_of_samples,?,?)
        """
        data_arr = np.array(hdf5_data.get('data'))
        label_arr = np.array(hdf5_data.get('label'))
        return data_arr.squeeze(), label_arr.squeeze()
