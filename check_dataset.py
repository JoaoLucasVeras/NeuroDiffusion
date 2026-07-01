import torch
eeg = torch.load('datasets/eeg_5_95_std.pth')
print("Sample 0 keys:", eeg['dataset'][0].keys())
print("Sample 0 image type:", type(eeg['dataset'][0]['image']))
print("Sample 0 image value:", eeg['dataset'][0]['image'])
if isinstance(eeg.get('images', None), list):
    print("Images list length:", len(eeg['images']))
    print("Images list first 5:", eeg['images'][:5])
elif eeg.get('images', None) is not None:
    print("Images object type:", type(eeg['images']))
