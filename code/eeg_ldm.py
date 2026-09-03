import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy

# own code
from config import Config_Generative_Model
from dataset import  create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM
from eval_metrics import get_similarity_metric


def str2bool(v):
    """argparse type=bool is a trap: bool("False") is True. Parse the string properly."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('expected a boolean value, got %r' % v)


def wandb_init(config, output_path):
    # wandb.init( project='dreamdiffusion',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []
    
    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))     
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None, 
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list
               
def save_samples_npz(samples, dataset, config):
    """Persist raw generations so eval_report.py can score them against baselines."""
    gt = np.stack([np.asarray(img[0]) for img in samples])
    pred = np.stack([np.stack([np.asarray(c) for c in img[1:]]) for img in samples])
    # stored channel-first by the sampler; the metrics expect channel-last uint8
    gt = rearrange(gt, 'n c h w -> n h w c')
    pred = rearrange(pred, 'n k c h w -> n k h w c')

    base = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    idx = dataset.split_idx if hasattr(dataset, 'split_idx') else range(len(gt))
    labels = np.array([base.data[i]['label'] for i in list(idx)[:len(gt)]])
    synsets = np.array(base.labels)

    out = os.path.join(config.output_path, 'samples.npz')
    np.savez_compressed(out, gt=gt.astype(np.uint8), pred=pred.astype(np.uint8),
                        labels=labels, synsets=synsets)
    print('saved raw samples to %s  (gt=%s pred=%s)' % (out, gt.shape, pred.shape))
    return out


def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))

    gen_limit = getattr(config, 'generate_limit', None)
    if gen_limit is not None and gen_limit > 0:
        print('generating %d of %d test trials (generate_limit)'
              % (min(gen_limit, len(eeg_latents_dataset_test)), len(eeg_latents_dataset_test)))
    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW, limit=gen_limit)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path,
                            f'./test{sp_idx}-{copy_idx}.png'))

    save_samples_npz(samples, eeg_latents_dataset_test, config)

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    print('eval metrics:', metric_dict)


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(config.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, 

        transforms.Resize((512, 512)),
        channel_last
    ])
    if config.dataset == 'EEG':

        eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(
                eeg_signals_path=config.eeg_signals_path, splits_path=config.splits_path,
                imagenet_path=getattr(config, 'imagenet_path', None),
                image_transform=[img_transform_train, img_transform_test],
                subject=config.subject,
                strict_images=getattr(config, 'strict_images', True))
        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_viz( image_transform=[img_transform_train, img_transform_test])
        num_voxels = eeg_latents_dataset_train.data_len

    else:
        raise NotImplementedError
    # print(num_voxels)

    # prepare pretrained mbm 

    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')

    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger, 
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune)
    
    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')
    # finetune the model
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=20)
    
    # Custom Callback for clear epoch tracking
    from pytorch_lightning.callbacks import Callback
    class PrintEpochCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            print(f"\n🚀 >>> STARTING EPOCH {trainer.current_epoch}/{trainer.max_epochs} <<< 🚀\n")

    # Enable periodic saving and clear printing
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_path, 'checkpoints'),
        filename='checkpoint-{epoch:02d}',
        every_n_epochs=50,
        save_top_k=-1
    )
    trainer.callbacks.extend([PrintEpochCallback(), checkpoint_callback])

    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config)

    # generate images
    # generate limited train images and generate images for subjects seperately
    generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)

    return

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default = '../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--eeg_signals_path', type=str)
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--imagenet_path', type=str)
    parser.add_argument('--subject', type=int)
    parser.add_argument('--clip_tune', type=str2bool)
    parser.add_argument('--cls_tune', type=str2bool)
    parser.add_argument('--cfg_scale', type=float)
    parser.add_argument('--generate_limit', type=int,
                        help='cap test trials sampled at the end of training; the full set '
                             'can take ~6h and runs after training, risking the walltime')
    parser.add_argument('--strict_images', type=str2bool, default=True,
                        help='abort if stimulus images are missing instead of '
                             'silently training against a blank target')

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=str2bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=str2bool)
    parser.add_argument('--eval_avg', type=str2bool)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


from pytorch_lightning.plugins.environments import SLURMEnvironment
def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count()
    # argparse gives precision as a string, but Lightning 1.6 only accepts an int for
    # numeric precisions -- "16" raises RuntimeError("No precision set"). Only "bf16"
    # is legitimately a string.
    if isinstance(precision, str) and precision.strip().isdigit():
        precision = int(precision)
    if num_gpus > 1:
        # Multi-GPU: use DDP
        return pl.Trainer(accelerator=acc, devices=num_gpus, strategy='ddp',
                plugins=[SLURMEnvironment(auto_requeue=True)], max_epochs=num_epoch, logger=logger,
                precision=precision, accumulate_grad_batches=accumulate_grad_batches,
                enable_checkpointing=True, enable_model_summary=False, gradient_clip_val=0.5,
                check_val_every_n_epoch=check_val_every_n_epoch)
    else:
        # Single GPU: let PL auto-select strategy (avoids DDP NCCL OOM)
        return pl.Trainer(accelerator=acc, devices=1, max_epochs=num_epoch, logger=logger,
                precision=precision, accumulate_grad_batches=accumulate_grad_batches,
                enable_checkpointing=True, enable_model_summary=False, gradient_clip_val=0.5,
                check_val_every_n_epoch=check_val_every_n_epoch)

  
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)
    
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    # Use root_path for results to ensure they appear in the main project folder
    output_path = os.path.join(config.root_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)
    
    wandb_init(config, output_path)

    logger = WandbLogger(project='dreamdiffusion', name=f"stage2-{datetime.datetime.now().strftime('%m%d-%H%M')}")
    config.logger = logger
    main(config)
