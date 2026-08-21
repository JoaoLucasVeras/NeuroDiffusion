import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 2.5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 10
        self.warmup_epochs = 2
        self.batch_size = 8
        self.clip_grad = 0.8
        
        # Model Parameters
        self.mask_ratio = 0.1
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '/home/015555345/NeruoDiffusion/'
        self.output_path = '../exps/'
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # Dataset paths. Built by code/prepare_shimizu_data.py, split by code/make_splits.py.
        # The old imagination_splits.pth was a random shuffle over sliding windows of the
        # same recordings -- 100% of its test recordings also appeared in train.
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/imagination_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/imagination_5_95_std_splits_subject_avail.pth')
        self.imagenet_path = os.path.join(self.root_path, 'datasets/imageNet_images')

        # distributed training
        self.local_rank = 0


class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):
        
        # Project setting
        self.root_path = '/home/015555345/NeruoDiffusion/'
        # self.root_path = '.'
        self.output_path = '../exps/'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')

        self.dataset = 'EEG' 
        self.pretrain_mbm_path = '../dreamdiffusion/pretrains/eeg_pretrain/checkpoint.pth' 

        self.include_nonavg_test = True


        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4 
        self.mask_ratio = 0.5
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '/home/015555345/NeruoDiffusion/'
        self.output_path = '../exps/'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/imagination_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/imagination_5_95_std_splits_subject_avail.pth')
        self.imagenet_path = os.path.join(self.root_path, 'datasets/imageNet_images')
        # Abort rather than silently substituting a blank target for a missing stimulus.
        self.strict_images = True
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 10
        self.lr = 5.3e-5
        self.num_epoch = 200
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 0
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # Sampling the whole test set after training costs hours and can be killed by the
        # walltime. 200 trials is plenty for tight CLIP error bars. None = no cap.
        self.generate_limit = 200
        self.cfg_scale = 8.0 # Critical for Imagination Paradigm
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 



class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '/home/015555345/NeruoDiffusion/'
        self.output_path = '../exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 50
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.15
        self.global_pool = False
        self.use_time_cond = False
        self.clip_tune = False
        self.subject = 1
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 
