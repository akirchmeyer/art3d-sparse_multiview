import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torchvision import transforms
import numpy as np
import itertools

from types import SimpleNamespace
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.cross_attention import CrossAttention
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14
import timm
        
from ptp_utils import AttentionStore, aggregate_attention, register_attention_control
from PIL import Image

def read_metadata(meta_data_file, img_data_path, depth_data_path, view_data_path, mask_data_path, subsample_inst_size):
    meta_df = pd.read_csv(meta_data_file, dtype={'index': str})
    inst_ids = {}
    inst_data = []
    meta_df = meta_df.sample(frac = 1) # shuffle
    for _, row in meta_df.iterrows():
        index = row['filename'].split('/')[1][:-4] # TODO: fix hack
        inst, view = [int(x) for x in index.split('.')]
        if not (inst in inst_ids):
            if subsample_inst_size < 0 or len(inst_data) < subsample_inst_size: 
                inst_ids[inst] = len(inst_data)
                inst_data.append({
                    'mask': f'{mask_data_path}/{inst}.npy', 
                    'image':f'{img_data_path}/{inst}.png', 
                    'depth':f'{depth_data_path}/{inst}.npy', 
                    'views':[],
                })
            else:
                continue
        inst_id = inst_ids[inst]
        img_data = {
            'image':f'{view_data_path}/{inst}.{view}.npy', 
            'elev':row['elev'], 
            'azim':row['azim']
        }
        inst_data[inst_id]['views'].append(img_data)
    return inst_data

def adjust_image_bbox(mask):
    mask = mask.numpy()
    idxs = np.argwhere(mask == 1)
    wh = np.array(mask.shape, dtype=float)
    min_xy, max_xy = idxs.min(axis=0), idxs.max(axis=0)
    ctr_xy, ext_xy = (min_xy+max_xy)/2, (max_xy-min_xy+1)/wh
    scale = ext_xy.max()
    ext_xy = scale * wh / 2
    min_xy, max_xy = np.maximum(ctr_xy - ext_xy, 0), np.minimum(ctr_xy + ext_xy + 1, wh) 

    return (min_xy, max_xy), (wh*(1-scale), wh*scale)

def crop_image(img, min_xy, max_xy):
    return img[..., int(min_xy[0]):int(max_xy[0]), int(min_xy[1]):int(max_xy[1])]

# See dreambooth train_dreambooth.py
class MultiViewDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, args, mode='train'):
        opts = args.dataset
        self.subsample_inst_size = opts[f'{mode}_subsample_inst_size']
        self.size = opts.resolution
        self.center_crop = opts.center_crop
        self.context_size = opts.context_resolution
        self.random_flip = opts.random_flip

        if not Path(opts.img_data_path).exists():
            raise ValueError(f"Instance {opts.img_data_path} images root doesn't exist.")
        if not Path(opts.meta_data_file).exists():
            raise ValueError(f"Instance {opts.meta_data_file} meta data file doesn't exist.")
        if not Path(opts.mask_data_path).exists():
            raise ValueError(f"Instance {opts.mask_data_path} meta data file doesn't exist.")

        # load data
        self.inst_data = read_metadata(opts.meta_data_file, opts.img_data_path, opts.depth_data_path, opts.view_data_path, opts.mask_data_path, self.subsample_inst_size)
        self.num_views_per_inst = opts.num_views_per_inst
        self.num_instances = len(self.inst_data)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, index):
        inst_id = index % self.num_instances
        inst_data = self.inst_data[inst_id]

        # depth + mask
        depth = torch.Tensor(np.load(inst_data['depth'])).squeeze(0)
        mask = torch.Tensor(np.load(inst_data['mask'])).bool()

        # crop
        #images = [crop_image_bbox(image) for image in images] # crop images
        # image


        center_crop, crop_only = adjust_image_bbox(mask)

        image = Image.open(inst_data['image']).convert("RGB")
        image = torch.Tensor(np.array(image)).unsqueeze(0).permute((0,3,1,2))/255
        image[:, :, mask==0] = 1 # mask in white
        image = crop_image(image, *center_crop) # center and crop 
        image = F.interpolate(image, size=self.size, mode='bilinear').squeeze(0)
        image = 2*image.contiguous()-1

        depth[mask==0] = 0 
        depth = crop_image(depth, *center_crop) # center and crop 
        depth = F.interpolate(depth.view(1,1,*depth.shape), size=self.size, mode='bilinear').squeeze()

        mask = crop_image(mask, *center_crop) # center and crop 
        mask = F.interpolate(mask.view(1,1,*mask.shape).float(), size=self.size, mode='bilinear').squeeze()
        mask = mask > 0
        # views
        views_id = list(range(len(inst_data['views'])))
        # sample views
        views_id = np.random.choice(views_id, size=self.num_views_per_inst, replace=False) # TODO: replace=True? mask?
        views_data = [inst_data['views'][i] for i in views_id]
        
        views_vp = torch.stack([torch.Tensor([view['elev'], view['azim']]) for view in views_data])
        # got OOM errors when using torchvision.transforms
        views_images = [np.load(view['image']) for view in views_data]
        views_images = torch.Tensor(np.array(views_images)).permute((0,3,1,2))/255 
        #views_images = crop_image(views_images, *crop_only) # center and crop 
        views_images = F.interpolate(views_images, size=self.context_size, mode='bilinear') 
        #views_images = 2*views_images.contiguous()-1

        # TODO: add positional encoding for viewpoints (MAE)
        return {'image': image, 'depth': depth, 'mask': mask, 'views': views_images, 'viewpoints': views_vp}
    
# TODO: add fp16
# See facebook MAE 
# See timm ViT
class MultiViewEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        opts = args.multiview_encoder
        self.model_name = opts.model_name
        match opts.model_name:
            case 'mae':
                self.encode_context = self.mae_encode_context
                self.mae_image_mask_ratio = opts.mae.image_mask_ratio
                match opts.model_size:
                    case 'base': self.model = mae_vit_base_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'large': self.model = mae_vit_large_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'huge': self.model = mae_vit_huge_patch14(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case _: raise "Unrecognized MAE model size"
            case 'vit':
                self.encode_context = self.vit_encode_context
                match opts.model_size:
                    case 'base': self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
                    case 'large': self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
                    case 'huge': self.model = timm.create_model('vit_huge_patch14_224', pretrained=True)
                    case _: raise "Unrecognized ViT model size"
            case _: raise "Unrecognized model name"

        # TODO: view mask ratio --> need viewpoint PE
        self.view_mask_ratio = opts.view_mask_ratio

    def mae_encode_context(self, context):
        #_, image, _ = self.mae(image, mask_ratio=self.image_mask_ratio)
        context, _, _ = self.model.forward_encoder(context.view(-1, *context.shape[2:]), mask_ratio=self.mae_image_mask_ratio)
        return context

    def vit_encode_context(self, context):
        context = self.model.forward_features(context.view(-1, *context.shape[2:]))
        return context

    def forward(self, input):
        # x: (N, C, H, W), views: (N, V, C, H, W) , V = n_views_per_inst
        image, context = input['image'], input['views']
        # TODO: add reconstruction loss from MAE?
        # TODO: MAE only accepts 224x224
        bsz = context.shape[0]
        # context: (bsz, num_views_per_inst, num_patches, hidden)
        context = self.encode_context(context)
        # context: (bsz, num_views_per_inst*num_patches, hidden)
        context = context.view(bsz, -1, context.shape[-1])
        return {'image': image, 'context': context}

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel

# See diffusers train_dreambooth.py
class MultiViewDiffusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load scheduler and models
        opts = args.diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(opts.sd_model)
        self.noise_scheduler = DDPMScheduler.from_pretrained(opts.sd_model, subfolder="scheduler")
        self.multiview_encoder = MultiViewEncoder(args)
        self.vae = self.pipe.vae
        self.unet = UNet2DConditionModel.from_pretrained(opts.sd_model, subfolder="unet")
        self.unet_uncond = UNet2DConditionModel.from_pretrained(opts.sd_model, subfolder="unet")
        self.train_multiview_encoder = opts.train_multiview_encoder
        self.with_prior_preservation = opts.with_prior_preservation
        self.prior_loss_weight = opts.prior_loss_weight if self.with_prior_preservation else None

        # reset unet crossattention params
        @torch.no_grad()
        def reset_crossattention_params(model):
            if model.__class__.__name__ == 'CrossAttention':
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        self.unet.apply(reset_crossattention_params)

        self.patch_sd_unet()
        # https://huggingface.co/docs/diffusers/optimization/fp16
        self.vae.enable_slicing()

    def finetunable_parameters(self):
        return itertools.chain(self.unet.parameters(), self.multiview_encoder.parameters()) if self.train_multiview_encoder else self.unet.parameters()
    
    def freeze_params(self):
        self.vae.requires_grad_(False)

        for name, params in self.unet.named_parameters():
            params.requires_grad = 'transformer_blocks' in name and ('attn2.to_k' in name or 'attn2.to_v' in name)

        if not self.train_multiview_encoder:
            self.multiview_encoder.requires_grad_(False)

    def finetune(self):
        self.unet.train()
        if self.train_multiview_encoder:
            self.multiview_encoder.train()

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        if self.train_multiview_encoder:
            self.multiview_encoder.gradient_checkpointing_enable()

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()

    def compute_loss(self, batch):
        # Get the text embedding for conditioning
        batch = self.multiview_encoder(batch)

        # Convert images to latent space
        latents = self.vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, batch['context']).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss


    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/run.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/notebooks/explain.ipynb
    def forward_with_crossattention(self, batch, res=16):
        controller = AttentionStore()
        cross_att_count = register_attention_control(self.unet, controller)
        result = self.forward(batch)
        attention_maps = aggregate_attention(attention_store=controller, res=res, from_where=("up", "down", "mid"), is_cross=True, select=0)
        cross_att_count2 = register_attention_control(self.unet, controller, unregister=True)
        assert(cross_att_count == cross_att_count2)
        return result, attention_maps

    def patch_sd_unet(self):
        old_encode_prompt = self.pipe._encode_prompt
        def new_encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None):
            assert(prompt is None and prompt_embeds is not None and negative_prompt is None and negative_prompt_embeds is None)
            batch_size = prompt_embeds.shape[0]
            uncond_prompt =  [""] * batch_size
            hidden_uncond = old_encode_prompt(uncond_prompt, device, num_images_per_prompt, do_classifier_free_guidance=False) if do_classifier_free_guidance else None
            hidden_cond   = old_encode_prompt(None, device, num_images_per_prompt, do_classifier_free_guidance=False, prompt_embeds=prompt_embeds)
            return SimpleNamespace(dtype=hidden_cond.dtype, uncond=hidden_uncond, cond=hidden_cond, do_classifier_free_guidance=do_classifier_free_guidance)

        def new_unet_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
            do_classifier_free_guidance = encoder_hidden_states.do_classifier_free_guidance

            if do_classifier_free_guidance:
                sample_uncond, sample_cond = sample.chunk(2)
                hidden_uncond, hidden_cond = encoder_hidden_states.uncond, encoder_hidden_states.cond
                
                sample = torch.cat([
                    self.unet_uncond(sample_uncond, timestep, hidden_uncond, *args, **kwargs).sample, 
                    self.unet(sample_cond, timestep, hidden_cond, *args, **kwargs).sample
                ], axis=0)
                return SimpleNamespace(sample=sample)
            else:
                # cond only  
                return self.unet(sample, timestep, encoder_hidden_states.cond,   *args, **kwargs)
        # no self passed to monkey-patched methods
        # https://stackoverflow.com/questions/28127874/monkey-patching-python-an-instance-method
        self.pipe._encode_prompt = new_encode_prompt
        self.pipe.unet.forward = new_unet_forward

    # see https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    #TODO: fix unconditional guidance
    def forward(self, batch):
        with torch.inference_mode():
            batch = self.multiview_encoder(batch)
            self.pipe.to(batch['context'].device)
            return self.pipe(prompt=None, prompt_embeds=batch['context']).images
