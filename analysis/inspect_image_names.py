"""
Run this on the HPC login node to inspect how image names are stored in the .mat file
and what the actual filenames look like when properly decoded.

Usage:
    python analysis/inspect_image_names.py
"""
import h5py
import numpy as np
import os

MAT_FILE = "datasets/Imagination Experiment/Imagine_Sub001.mat"
IMAGENET_DIR = "datasets/imageNet_images"

print("="*60)
print("Inspecting image references in MAT file")
print("="*60)

with h5py.File(MAT_FILE, 'r') as f:
    images_raw = f['images']
    print(f"\nImages dataset shape: {images_raw.shape}")
    print(f"Images dataset dtype: {images_raw.dtype}")

    # Check if it's HDF5 object references (MATLAB cell array of strings)
    if images_raw.dtype == object:
        print("\n→ Detected HDF5 object references (MATLAB cell array of strings)")
        print("\nDecoding first 10 image filenames:")
        for i, ref in enumerate(images_raw.flatten()[:10]):
            try:
                name = ''.join(chr(c) for c in f[ref][:].flatten())
                print(f"  [{i}] → '{name}'")
            except Exception as e:
                print(f"  [{i}] → ERROR: {e}")
    else:
        print(f"\n→ Raw numeric values (NOT object references)")
        print(f"First 10 values: {images_raw[:10].flatten()}")
        print(f"Min: {images_raw[:].min()}, Max: {images_raw[:].max()}")

# Check what actual files exist in ImageNet folder
print("\n" + "="*60)
print("Checking actual ImageNet folder contents")
print("="*60)

synsets = sorted(os.listdir(IMAGENET_DIR)) if os.path.exists(IMAGENET_DIR) else []
print(f"\nNumber of synset folders: {len(synsets)}")
if synsets:
    first_synset = synsets[0]
    folder = os.path.join(IMAGENET_DIR, first_synset)
    files = sorted(os.listdir(folder))[:5]
    print(f"\nFirst synset: {first_synset}")
    print(f"Sample files: {files}")
