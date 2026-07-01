import torch
eeg = torch.load('datasets/eeg_5_95_std.pth')
print("Total images in self.images:", len(eeg['images']))
print("First 10 images:", eeg['images'][:10])
