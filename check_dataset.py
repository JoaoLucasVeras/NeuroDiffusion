import torch
eeg = torch.load('datasets/eeg_5_95_std.pth')
print("Keys in eeg:", eeg.keys())
