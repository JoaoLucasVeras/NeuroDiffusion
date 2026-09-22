import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import eeg_encoder, classify_network, mapping 
from PIL import Image
def create_model_from_config(config, num_voxels, global_pool):
    model = eeg_encoder(time_len=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
    return model

def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

class cond_stage_model(nn.Module):
    def __init__(self, metafile, num_voxels=440, cond_dim=1280, global_pool=True, clip_tune = True, cls_tune = False):
        super().__init__()
        # prepare pretrained fmri mae 
        if metafile is not None:
            model = create_model_from_config(metafile['config'], num_voxels, global_pool)
        
            model.load_checkpoint(metafile['model'])
        else:
            model = eeg_encoder(time_len=num_voxels, global_pool=global_pool)
        self.mae = model
        if clip_tune:
            self.mapping = mapping()
            # CLIP-style learnable temperature, init 1/0.07 (only used by 'contrastive').
            self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1 / 0.07)))))
        if cls_tune:
            self.cls_net = classify_network()

        self.fmri_seq_len = model.num_patches
        self.fmri_latent_dim = model.embed_dim
        if global_pool == False:
            self.channel_mapper = nn.Sequential(
                nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True),
                nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True)
            )
        self.dim_mapper = nn.Linear(self.fmri_latent_dim, cond_dim, bias=True)
        self.global_pool = global_pool

        # self.image_embedder = FrozenImageEmbedder()

    # def forward(self, x):
    #     # n, c, w = x.shape
    #     latent_crossattn = self.mae(x)
    #     if self.global_pool == False:
    #         latent_crossattn = self.channel_mapper(latent_crossattn)
    #     latent_crossattn = self.dim_mapper(latent_crossattn)
    #     out = latent_crossattn
    #     return out

    def forward(self, x):
        # n, c, w = x.shape
        # Cast input to match model dtype. model.half() puts weights in fp16 but
        # DataLoader returns fp32. With precision=32 (no AMP), Lightning won't
        # auto-cast, so we do it explicitly here to avoid dtype mismatch in Conv1d.
        x = x.to(next(self.mae.parameters()).dtype)
        latent_crossattn = self.mae(x)
        latent_return = latent_crossattn
        if self.global_pool == False:
            latent_crossattn = self.channel_mapper(latent_crossattn)
        latent_crossattn = self.dim_mapper(latent_crossattn)
        out = latent_crossattn
        return out, latent_return

    # def recon(self, x):
    #     recon = self.decoder(x)
    #     return recon

    def get_cls(self, x):
        return self.cls_net(x)

    def get_clip_loss(self, x, image_embeds, stems=None):
        """EEG->CLIP alignment loss.

        'cosine'      : the original pointwise 1 - cos(pred, target). With ~33 training
                        stimuli a large encoder drives this to ~0 by memorising which
                        target each window belongs to (observed: 0.22 -> 1e-4).
        'contrastive' : symmetric InfoNCE over the batch with a learnable temperature.
                        Windows of the same stimulus (same stem) are treated as
                        positives of each other, so they are never pushed apart.
        """
        target_emb = self.mapping(x)
        if getattr(self, 'clip_loss_type', 'cosine') != 'contrastive':
            return 1 - torch.cosine_similarity(target_emb, image_embeds, dim=-1).mean()

        pn = F.normalize(target_emb.float(), dim=-1)
        tn = F.normalize(image_embeds.float(), dim=-1)
        logits = pn @ tn.T * self.logit_scale.exp().clamp(max=100)
        n = logits.shape[0]
        if stems is not None and len(stems) == n:
            ids = {}
            key = torch.tensor([ids.setdefault(st, len(ids)) for st in stems], device=logits.device)
            pos = (key[:, None] == key[None, :]).float()
        else:
            pos = torch.eye(n, device=logits.device)

        def sup_con(lg):
            logp = F.log_softmax(lg, dim=1)
            return -(torch.logsumexp(logp + torch.log(pos.clamp_min(1e-12)), dim=1)).mean()

        return 0.5 * (sup_con(logits) + sup_con(logits.T))
    


class eLDM:

    def __init__(self, metafile, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        # self.ckp_path = os.path.join(pretrain_root, 'model.ckpt')
        self.ckp_path = os.path.join(pretrain_root, 'models/v1-5-pruned.ckpt')
        self.config_path = os.path.join(pretrain_root, 'models/config15.yaml') 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(metafile, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        # NOTE: model.half() is NOT called here. On the A100 (40GB), the fp32 model
        # fits comfortably. Calling model.half() makes gradients fp16, which causes
        # AMP's GradScaler to raise: ValueError: Attempting to unscale FP16 gradients.
        # AMP (--precision 16) handles fp16 activations automatically with fp32 params.
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim
        self.metafile = metafile

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        # Validation is carved out of the TRAINING subjects; the test split is only
        # logged. Selecting a checkpoint on the test subject would be model selection
        # on the test set.
        from dataset import split_val_from_train
        n_val_w = int(getattr(config, 'val_windows', 4))
        if n_val_w > 0:
            dataset, val_dataset = split_val_from_train(dataset, test_dataset, n_val_windows=n_val_w,
                                                        gap=int(getattr(config, 'val_gap', 1)))
        else:
            val_dataset = test_dataset
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=bs1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        # Regularisation knobs (see IMPROVEMENT_PLAN.md, P1/P2).
        self.model.weight_decay = float(getattr(config, 'weight_decay', 0.01))
        self.model.clip_weight = float(getattr(config, 'clip_weight', 1.0))
        self.model.val_preview_every = int(getattr(config, 'val_preview_every', 0))
        self.model.cond_stage_model.clip_loss_type = getattr(config, 'clip_loss', 'cosine')
        n_freeze = int(getattr(config, 'freeze_encoder_blocks', 0))
        enc = self.model.cond_stage_model.mae
        frozen = set()
        if n_freeze > 0:
            mods = [enc.patch_embed] + list(enc.blocks[:n_freeze])
            for m in mods:
                for p in m.parameters():
                    frozen.add(id(p))
            print('freezing patch_embed + %d/%d encoder blocks (%d tensors)'
                  % (n_freeze, len(enc.blocks), len(frozen)))
        self.model.frozen_params = frozen

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg

        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        monitor = getattr(config, 'select_metric', 'val/retrieval_top1')
        best_cb = ModelCheckpoint(dirpath=os.path.join(output_path, 'checkpoints'),
                                  filename='best-{epoch:03d}', monitor=monitor, mode='max',
                                  save_top_k=1, save_weights_only=True)
        cbs = [best_cb]
        patience = int(getattr(config, 'early_stop_patience', 0))
        if patience > 0:
            cbs.append(EarlyStopping(monitor=monitor, mode='max', patience=patience, verbose=True))
        trainers.callbacks.extend(cbs)

        trainers.fit(self.model, dataloader, val_dataloaders=[val_loader, test_loader])

        # checkpoint.pth = best held-out checkpoint (what eval scores); keep last too.
        torch.save({'model_state_dict': self.model.state_dict(), 'config': config,
                    'state': torch.random.get_rng_state()},
                   os.path.join(output_path, 'checkpoint_last.pth'))
        if best_cb.best_model_path and os.path.exists(best_cb.best_model_path):
            sd = torch.load(best_cb.best_model_path, map_location='cpu')['state_dict']
            self.model.load_state_dict(sd, strict=False)
            print('restored best checkpoint %s (%s=%.4f)'
                  % (best_cb.best_model_path, monitor, float(best_cb.best_model_score or float('nan'))))
        else:
            print('WARNING: no best checkpoint recorded; checkpoint.pth is the last epoch')

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            # Checkpoints store the CPU generator state (torch.random.get_rng_state),
            # so restore it there; loading it into the CUDA generator raises
            # "RNG state is wrong size". Seed CUDA from it for reproducibility.
            torch.random.set_rng_state(state)
            torch.cuda.manual_seed_all(int(torch.randint(0, 2**31 - 1, (1,))))
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                
                # CFG null conditioning using zeros to push the model against a baseline noise
                uc = model.get_learned_conditioning(torch.zeros_like(repeat(latent, 'h w -> c h w', c=num_samples)).to(self.device))[0]

                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                unconditional_guidance_scale=getattr(model.main_config, 'cfg_scale', 8.0) if hasattr(model, 'main_config') else 8.0,
                                                unconditional_conditioning=uc,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)




class eLDM_eval:

    def __init__(self, config_path, num_voxels, device=torch.device('cpu'),
                 pretrain_root='../pretrains/',
                 logger=None, ddim_steps=250, global_pool=True, use_time_cond=False, clip_tune = True, cls_tune = False):
        self.config_path = config_path # 
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim

        model = instantiate_from_config(config.model)

        model.cond_stage_trainable = True
        model.cond_stage_model = cond_stage_model(None, num_voxels, self.cond_dim, global_pool=global_pool, clip_tune = clip_tune,cls_tune = cls_tune)

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        # Keep model in fp32 for stability with BF16/AMP.
        # Batch size is lowered to 2 to ensure we stay within 40GB during sampling.
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        
        self.device = device    
        self.model = model
        
        self.model.clip_tune = clip_tune
        self.model.cls_tune = cls_tune

        self.ldm_config = config
        self.pretrain_root = pretrain_root
        self.fmri_latent_dim = model.cond_stage_model.fmri_latent_dim

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=bs1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()
        # self.model.freeze_whole_model()
        # self.model.unfreeze_cond_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, fmri_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None, output_path = None):
        # fmri_embedding: n, seq_len, embed_dim
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            # Checkpoints store the CPU generator state (torch.random.get_rng_state),
            # so restore it there; loading it into the CUDA generator raises
            # "RNG state is wrong size". Seed CUDA from it for reproducibility.
            torch.random.set_rng_state(state)
            torch.cuda.manual_seed_all(int(torch.randint(0, 2**31 - 1, (1,))))
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(fmri_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                # print(item)
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                # assert latent.shape[-1] == self.fmri_latent_dim, 'dim error'
                
                c, re_latent = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                
                # CFG null conditioning using zeros to push the model against a baseline noise
                uc = model.get_learned_conditioning(torch.zeros_like(repeat(latent, 'h w -> c h w', c=num_samples)).to(self.device))[0]

                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                unconditional_guidance_scale=getattr(model.main_config, 'cfg_scale', 8.0) if hasattr(model, 'main_config') else 8.0,
                                                unconditional_conditioning=uc,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                if output_path is not None:
                    samples_t = (255. * torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0).numpy()).astype(np.uint8)
                    for copy_idx, img_t in enumerate(samples_t):
                        img_t = rearrange(img_t, 'c h w -> h w c')
                        Image.fromarray(img_t).save(os.path.join(output_path, 
                            f'./test{count}-{copy_idx}.png'))
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)
