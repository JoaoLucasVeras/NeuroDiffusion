import torch

eeg_path = 'datasets/eeg_5_95_std.pth'
splits_path = 'datasets/block_splits_by_image_single.pth'

print("Loading EEG data...")
eeg = torch.load(eeg_path)
print("EEG dataset total samples:", len(eeg['dataset']))

# Count samples per subject
subjects = {}
for i, item in enumerate(eeg['dataset']):
    sub = item.get('subject', 'UNKNOWN')
    subjects[sub] = subjects.get(sub, 0) + 1
print("Samples per subject:", subjects)

print("\nLoading splits...")
splits = torch.load(splits_path)
print("Splits keys:", splits.keys())
if 'splits' in splits:
    print("Number of splits:", len(splits['splits']))
    for i, sp in enumerate(splits['splits']):
        print(f"Split {i} keys:", sp.keys())
        if 'train' in sp:
            train_idx = sp['train']
            print(f"  Split {i} Train max index: {max(train_idx) if train_idx else 'EMPTY'}, len: {len(train_idx)}")
            if len(train_idx) > 0:
                print(f"  Split {i} Train first 5 indices:", train_idx[:5])
