import scipy.io as sio
import numpy as np
import os

# Point this to the first subject in your new folder structure
# Based on my search, the file is at: datasets\Imagination Experiment\Imagine_Sub001.mat
file_path = os.path.join('datasets', 'Imagination Experiment', 'Imagine_Sub001.mat')

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    print(f"Inspecting file: {file_path}\n")

    # Load the MATLAB file
    try:
        # Try scipy first
        try:
            mat_dict = sio.loadmat(file_path)
            print("--- MATLAB (v7 or older) Dictionary Keys & Array Shapes ---")
            for key, value in mat_dict.items():
                if not key.startswith('__'):
                    if isinstance(value, np.ndarray):
                        print(f"Variable: '{key}' | Type: Array | Shape: {value.shape}")
                    else:
                        print(f"Variable: '{key}' | Type: {type(value)}")
        except NotImplementedError:
            # This happens for MATLAB v7.3 files
            import h5py
            with h5py.File(file_path, 'r') as f:
                print("--- MATLAB (v7.3/HDF5) Dictionary Keys & Array Shapes ---")
                for key in f.keys():
                    value = f[key]
                    if isinstance(value, h5py.Dataset):
                        print(f"Variable: '{key}' | Type: Array/Dataset | Shape: {value.shape}")
                    else:
                        print(f"Variable: '{key}' | Type: {type(value)}")

    except Exception as e:
        print(f"Error loading .mat file: {e}")
