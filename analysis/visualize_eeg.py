import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_eeg(pth_path, output_dir='analysis/visualizations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Loading {pth_path}...")
    data = torch.load(pth_path, map_location='cpu')
    
    # Extract some samples
    samples = data['dataset']
    num_samples_to_plot = 3
    
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(15, 5 * num_samples_to_plot))
    
    for i in range(num_samples_to_plot):
        sample = samples[i]
        eeg = sample['eeg'].numpy() # (128, 125)
        label = sample['label']
        subject = sample['subject']
        image_name = sample['image']
        
        # Plot 1: Heatmap of all channels
        im = axes[i, 0].imshow(eeg, aspect='auto', cmap='viridis')
        axes[i, 0].set_title(f"Sample {i}: Heatmap (Subj: {subject}, Label: {label})\nImage: {image_name}")
        axes[i, 0].set_ylabel("Channels (128)")
        axes[i, 0].set_xlabel("Time points (125)")
        fig.colorbar(im, ax=axes[i, 0])
        
        # Plot 2: Selection of channels as time series
        # Pick 5 channels evenly spaced
        channels_to_plot = np.linspace(0, 127, 5, dtype=int)
        for ch in channels_to_plot:
            axes[i, 1].plot(eeg[ch], label=f'Ch {ch}')
        
        axes[i, 1].set_title(f"Sample {i}: Time Series (Selected Channels)")
        axes[i, 1].set_xlabel("Time points")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].legend(loc='upper right', fontsize='small')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'eeg_visualization.png')
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_eeg(r'datasets/eeg_5_95_std.pth')
