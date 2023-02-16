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
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14
import timm
import os
        
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel

from multiview import MultiViewEncoder
from cross_attention.cross_attention import ResidualCrossAttention, CrossAttnProcessorPatch, aggregate_hidden_maps, patch_attention_control, patch_residual_attention, aggregate_attention
from transformers import AutoTokenizer

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
        self.unet = UNet2DConditionModel.from_pretrained(opts.sd_model, subfolder='unet')
        self.tokenizer = AutoTokenizer.from_pretrained(opts.sd_model, subfolder='tokenizer', use_fast=False)
        self.train_multiview_encoder = opts.train_multiview_encoder
        self.guidance_scale = opts.guidance_scale
        self.num_inference_steps = opts.num_inference_steps

        self.loss_weights = {k:v for k,v in opts['loss_weights'].items() if v is not None}
        self.loss_decays = {k:v for k,v in opts['loss_decays'].items() if v is not None}
        self.loss_decays  = {k:self.loss_decays.get(k, 1) for k in self.loss_weights.keys()}
        print('loss_weights:', self.loss_weights)
        print('loss_decays:',  self.loss_decays)
        
        self.unet_inv = UNet2DConditionModel.from_pretrained(opts.sd_model, subfolder='unet')
        patch_attention_control(self.unet_inv)
        self.unet_inv.requires_grad_(False)
        self.unet_inv.eval()

        patch_residual_attention(args, self.unet)

        # https://huggingface.co/docs/diffusers/optimization/fp16
        # self.vae.enable_slicing()

    def finetunable_parameters(self):
        return itertools.chain(self.unet.parameters(), self.multiview_encoder.parameters()) if self.train_multiview_encoder else self.unet.parameters()
    
    def freeze_params(self, verbose=True):
        self.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        for name, params in self.unet.named_parameters():
            params.requires_grad = 'transformer_blocks' in name and 'attn2' in name and 'multiview_cross' in name
        if not self.train_multiview_encoder:
            self.multiview_encoder.requires_grad_(False)

    def finetune(self, verbose=False):
        self.unet.train()
        if self.train_multiview_encoder:
            self.multiview_encoder.train()
        
        self.check_parameters(verbose=verbose)

    def check_parameters(self, verbose=True):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if verbose:
                    print(f'requires_grad: {name}')
                assert('multiview_cross' in name)

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        if self.train_multiview_encoder:
            self.multiview_encoder.gradient_checkpointing_enable()

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()

    def compute_loss(self, batch, epoch):
        # Get the text embedding for conditioning
        batch = self.multiview_encoder(batch)
        device = batch['target'].device
        self.pipe.to(device)

        # Convert images to latent space
        latents = self.vae.encode(batch["target"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image

        # TODO: remove -1 for ddim inversion
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps-1, (bsz,), device=device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        input_ids = batch['input_ids'] #self.pipe.tokenizer(batch['prompt'], truncation=True, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt").input_ids
        prompt_embeds = self.pipe.text_encoder(input_ids.to(device))[0]
        #prompt_embeds = self.pipe._encode_prompt(batch['prompt'], device, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=batch['neg_prompt'])
        
        model_pred = self.unet(noisy_latents, timesteps, prompt_embeds, cross_attention_kwargs=dict(multiview_hidden_states=batch['source'])).sample
        source_attention_maps = aggregate_hidden_maps(self.unet, "CrossAttention", "multiview_cross", step=-1, detach=False, clear=True, map_type='hidden')

        losses = {}
        if 'ddim_inv' in self.loss_weights:
            # add the cross attention map to the dictionary
            x_invs = torch.stack([batch['inv'][i,t,:,:,:] for i,t in enumerate(timesteps)])
            model_inv_pred = self.unet_inv(x_invs, timesteps, prompt_embeds).sample
            target_maps = aggregate_hidden_maps(self.unet_inv, "CrossAttention", "attn2", step=-1, detach=True, clear=True, map_type='hidden')

            ddim_loss = 0
            for key_tgt in target_maps.keys():
                key_src = f'{key_tgt}.multiview_cross'
                assert(key_src in source_attention_maps)
                ddim_loss += F.mse_loss(source_attention_maps[key_src].float(),target_maps[key_tgt].float(), reduction='mean')
            losses['ddim_inv'] = ddim_loss

        # Get the target for loss depending on the prediction type
        match self.noise_scheduler.config.prediction_type:
            case "epsilon":     target = noise
            case "v_prediction": target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            case _: raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        losses['denoise'] = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        #loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        #return loss, {'total': loss.detach().item()}

        loss = 0
        losses = {key:val * self.loss_weights[key] * (self.loss_decays[key] ** epoch) for key, val in losses.items()}
        for key, val in losses.items():
            loss += val
        losses = {k:v.detach().item() for k,v in losses.items()}
        return loss, {'total': loss.detach().item(), **losses}

    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/run.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/notebooks/explain.ipynb
    def extract_hidden_maps(self, filter_fn, map_type='attention', res=16, step=-1, clear=True):
        target_maps = aggregate_hidden_maps(self.unet_inv, "CrossAttention", "attn2", detach=True, step=step, clear=clear, map_type=map_type) 
        source_maps = aggregate_hidden_maps(self.unet, "CrossAttention", "multiview_cross", detach=True, step=step, clear=clear, map_type=map_type)
        target_attention_maps = aggregate_attention(target_maps, res=res, filter_fn=filter_fn, cond=True)
        source_attention_maps = aggregate_attention(source_maps, res=res, filter_fn=filter_fn, cond=True)
        return target_attention_maps, source_attention_maps

    # see https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    def forward(self, batch, latents=None):
        with torch.inference_mode():
            batch = self.multiview_encoder(batch)
            self.pipe.to(batch['source'].device) 
            
            self.pipe.unet = self.unet
            pred = self.pipe(
                latents=latents, 
                prompt=batch['prompt'], 
                negative_prompt=batch['neg_prompt'], 
                cross_attention_kwargs=dict(multiview_hidden_states=batch['source']), 
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale
            ).images[0]
            
            self.pipe.unet = self.unet_inv
            target = self.pipe(
                latents=batch['inv'][:,-1,:,:,:], 
                prompt=batch['prompt'], 
                negative_prompt=batch['neg_prompt'], 
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale).images[0]

        return {'pred': torch.Tensor(np.array(pred)).permute((2,0,1)), 'target': torch.Tensor(np.array(target)).permute((2,0,1)), 'input': batch}