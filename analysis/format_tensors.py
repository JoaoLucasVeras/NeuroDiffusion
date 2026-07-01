import h5py
import numpy as np
import torch
import os
import random

# 1. Define Paths
# Adjusted to match the actual path found in the repository
input_file = 'datasets/Imagination Experiment/Imagine_Sub001.mat'
output_dir = 'datasets/'
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_file):
    print(f"Error: File not found at {input_file}")
    exit(1)

print("Loading HDF5 MATLAB file...")
with h5py.File(input_file, 'r') as f:
    raw_data = np.array(f['data'])
    raw_labels = np.array(f['labels']).flatten()

# Ensure shape is exactly [800 chunks, 128 channels, 125 timepoints]
if raw_data.shape == (125, 128, 800):
    raw_data = np.transpose(raw_data, (2, 1, 0))
elif raw_data.shape == (800, 128, 125):
    pass
else:
    raw_data = raw_data.reshape(800, 128, 125)

print("Normalizing EEG voltages...")
means = np.mean(raw_data, axis=2, keepdims=True)
stds = np.std(raw_data, axis=2, keepdims=True)
normalized_data = (raw_data - means) / (stds + 1e-8) 

# Convert to PyTorch Tensor
eeg_tensor = torch.tensor(normalized_data, dtype=torch.float32)

# --- THE FIX: Wrap the tensor in the expected 'dataset' LIST of DICTIONARIES ---
# Baseline dataset.py expects: loaded['dataset'][i]['subject']
print("Converting to list of dictionaries for baseline compatibility...")
dataset_list = []
for i in range(len(eeg_tensor)):
    dataset_list.append({
        'eeg': eeg_tensor[i],
        'subject': 1, # Subject 001
        'label': int(raw_labels[i]),
        'image': f"img_{int(raw_labels[i])}_{i}.jpg" # Dummy image name
    })

dataset_dict = {
    'dataset': dataset_list,
    'labels': list(set(int(l) for l in raw_labels)),
    'images': [item['image'] for item in dataset_list]
}

print("Building nested dataset splits dictionary...")
# To prevent data leakage, we split by the 40 unique ImageNet classes
unique_labels = np.unique(raw_labels).tolist()
random.seed(42) # For reproducibility
random.shuffle(unique_labels)

# 80% Train (32 classes), 10% Val (4 classes), 10% Test (4 classes)
train_classes = unique_labels[:32]
val_classes = unique_labels[32:36]
test_classes = unique_labels[36:]

train_indices = []
val_indices = []
test_indices = []

# Map the 800 chunks to their respective split based on their class
for i, label in enumerate(raw_labels):
    if label in train_classes:
        train_indices.append(i)
    elif label in val_classes:
        val_indices.append(i)
    else:
        test_indices.append(i)

# Wrap it in the exact nested structure: loaded['splits'][0]['train']
splits_dict = {
    'splits': [{
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }]
}

# 4. Save to Disk
print("Saving PyTorch .pth files...")
torch.save(dataset_dict, os.path.join(output_dir, 'eeg_5_95_std.pth'))
torch.save(splits_dict, os.path.join(output_dir, 'block_splits_by_image_single.pth'))

# --- BONUSES: Stage 1 prep ---
# Stage 1 (stageA1_eeg_pretrain.py) expects .npy files in datasets/mne_data/
print("Generating .npy files for Stage 1 pretraining...")
mne_dir = os.path.join(output_dir, 'mne_data')
os.makedirs(mne_dir, exist_ok=True)
for i in range(len(normalized_data)):
    np.save(os.path.join(mne_dir, f"sub001_chunk_{i:03d}.npy"), normalized_data[i])

print("\nSUCCESS! Nested Tensors AND .npy fragments generated.")
print("You are now fully compatible with BOTH Stage 1 (MAE) and Stage 2 (Diffusion).")
