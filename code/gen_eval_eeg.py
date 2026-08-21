import os, sys
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse


from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root', type=str, default='../dreamdiffusion/')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to dataset splits.')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='Path to EEG signals data.')

    parser.add_argument('--config_patch', type=str, default=None,
                        help='sd config path.')
    
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='imagenet path.')

    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    root = args.root
    target = args.dataset

    sd = torch.load(args.model_path, map_location='cpu')
    if 'config' in sd:
        config = sd['config']
    else:
        print("⚠️ Warning: config not found in checkpoint. Using default Config_Generative_Model.")
        config = Config_Generative_Model()
    # update paths
    config.root_path = root


    output_path = os.path.join(config.root_path, 'results', 'eval',  
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)), 
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), 
        channel_last
    ])

    
    dataset_train, dataset_test = create_EEG_dataset(
                eeg_signals_path=args.eeg_signals_path,
                splits_path=args.splits_path, imagenet_path=args.imagenet_path,
                image_transform=[img_transform_train, img_transform_test],
                subject=config.subject,
                strict_images=getattr(config, 'strict_images', True))
    num_voxels = dataset_test.dataset.data_len



    # create generateive model
    generative_model = eLDM_eval(args.config_patch, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=getattr(config, 'logger', None),
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    # m, u = model.load_state_dict(pl_sd, strict=False)
    if 'model_state_dict' in sd:
        generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    elif 'state_dict' in sd:
        # Lightning format
        generative_model.model.load_state_dict(sd['state_dict'], strict=False)
    else:
        # Raw state dict
        generative_model.model.load_state_dict(sd, strict=False)
        
    print('load ldm successfully')
    state = sd.get('state', None)
    os.makedirs(output_path, exist_ok=True)
    grid, _ = generative_model.generate(dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

    grid, samples = generative_model.generate(dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, limit=None, state=state, output_path = output_path) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))


    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))

    # Persist raw generations so eval_report.py can score them against the
    # noise and shuffled-pairing baselines.
    gt = np.stack([np.asarray(img[0]) for img in samples])
    pred = np.stack([np.stack([np.asarray(c) for c in img[1:]]) for img in samples])
    gt = rearrange(gt, 'n c h w -> n h w c')
    pred = rearrange(pred, 'n k c h w -> n k h w c')
    base = dataset_test.dataset
    idx = list(dataset_test.split_idx)[:len(gt)]
    np.savez_compressed(os.path.join(output_path, 'samples.npz'),
                        gt=gt.astype(np.uint8), pred=pred.astype(np.uint8),
                        labels=np.array([base.data[i]['label'] for i in idx]),
                        synsets=np.array(base.labels))
    print('saved raw samples to', os.path.join(output_path, 'samples.npz'))
