import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
import os

from multiview import MultiViewEncoder
from controlnet.patched_unet import create_unet
from cross_attention import aggregate_attention, aggregate_activations, hook_activations

# See diffusers train_dreambooth.py
class MultiViewDiffusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        opts = args.diffusion

        # Parameters
        self.train_multiview_encoder = opts.train_multiview_encoder
        self.train_full_unet = opts.train_full_unet
        # Load subcomponents
        self.pipe = StableDiffusionPipeline.from_pretrained(opts.sd_model)
        self.pipe.scheduler = DDIMScheduler.from_pretrained(opts.sd_model, subfolder="scheduler")
        self.multiview_encoder = MultiViewEncoder(args)
        # self.vae = self.pipe.vae
        # self.tokenizer = AutoTokenizer.from_pretrained(opts.sd_model, subfolder='tokenizer', use_fast=False)
        # self.vae.enable_slicing() # https://huggingface.co/docs/diffusers/optimization/fp16

        self.setup_loss_weights(opts)
        self.setup_unet(args)
        self.setup_unet_inv(opts)
        self.freeze_params()

    def setup_loss_weights(self, opts):
        # Loss weights and decays
        self.loss_weights = {k:v for k,v in opts['loss_weights'].items() if v is not None}
        #self.loss_decays = {k:v for k,v in opts['loss_decays'].items() if v is not None}
        #self.loss_decays  = {k:self.loss_decays.get(k, 1) for k in self.loss_weights.keys()}
        print('loss_weights:', self.loss_weights)
        #print('loss_decays:',  self.loss_decays)
        
    def setup_unet_inv(self, opts):
        # Unet inverse
        self.unet_inv = UNet2DConditionModel.from_pretrained(opts.sd_model, subfolder='unet')
        hook_activations(self.unet_inv)

    def setup_unet(self, args):
        opts = args.diffusion
        self.patch_type = opts.patch_type
        unet = create_unet(opts, opts.sd_model, cross_attention_dim=self.multiview_encoder.total_cross_attention_dim)
        
        match self.patch_type:
            case 'crossattention': self.unet = unet
            case 'transformer_block': self.unet = unet
            case 'controlnet': self.unet, self.controlnet = unet
            case 'instructpix2pix': self.unet = unet
            case _: raise "unrecognized patch_type"

        hook_activations(self.unet)

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        if self.train_multiview_encoder:
            self.multiview_encoder.gradient_checkpointing_enable()

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()

    def finetunable_parameters(self):
        match self.patch_type:
            case 'crossattention': unet_params = self.unet.parameters()
            case 'transformer_block': unet_params = self.unet.parameters()
            case 'controlnet': unet_params = self.controlnet.parameters()
            case 'instructpix2pix': unet_params = self.unet.parameters()
            case _: raise "unrecognized patch_type"
        return itertools.chain(unet_params, self.multiview_encoder.finetunable_parameters()) if self.train_multiview_encoder else unet_params
    
    def freeze_params(self, verbose=True):
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.multiview_encoder.requires_grad_(self.train_multiview_encoder)
        self.unet_inv.requires_grad_(False)

        match self.patch_type:
            case 'crossattention':
                for name, params in self.unet.named_parameters():
                    params.requires_grad = 'multiview_cross' in name
            case 'transformer_block':
                for name, params in self.unet.named_parameters():
                    params.requires_grad = 'source_transformer' in name
            case 'controlnet':
                self.unet.requires_grad_(False)
                self.controlnet.requires_grad_(True)
            case 'instructpix2pix': 
                self.unet.requires_grad_(True)
            case _: raise "unrecognized patch_type"

        if self.train_full_unet:
            self.unet.requires_grad_(True)

    def check_parameters(self, verbose=True):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if verbose:
                    print(f'requires_grad: {name}')
                #assert('multiview_cross' in name)

    def finetune(self, verbose=False):
        match self.patch_type:
            case 'crossattention': self.unet.train()
            case 'transformer_block': self.unet.train()
            case 'controlnet': self.controlnet.train()
            case 'instructpix2pix': self.unet.train()
            case _: raise "unrecognized patch_type"

        if self.train_multiview_encoder:
            self.multiview_encoder.finetune()
        self.check_parameters(verbose=verbose)

    def denoise_one(self, batch):
        bsz, device = batch['target'].shape[0], batch['target'].device
        self.pipe.to(device)

        # Latent embedding + noise
        latents = self.pipe.vae.encode(batch["target"]).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        
        # Text embedding
        input_ids = self.pipe.tokenizer(batch['prompt'], truncation=True, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt").input_ids
        prompt_embeds = self.pipe.text_encoder(input_ids.to(device))[0]
        #prompt_embeds = self.pipe._encode_prompt(batch['prompt'], device, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=batch['neg_prompt'])
        
        multiview_embeds = self.multiview_encoder(batch)
        # Timesteps - NOTE: remove last timestep for ddim inversion
        timesteps = torch.randint(0, self.pipe.scheduler.config.num_train_timesteps-1, (bsz,), device=device).long()
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

        # predict
        match self.patch_type:
            case 'crossattention': 
                pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False, 
                    cross_attention_kwargs=dict(multiview_hidden_states=multiview_embeds))[0]
            case 'transformer_block': 
                pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False, 
                    cross_attention_kwargs=dict(multiview_hidden_states=multiview_embeds, x_t=noisy_latents))[0]
            case 'controlnet': 
                control = self.controlnet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, controlnet_hint=batch['source'], return_dict=False)
                pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, return_dict=False, control=control)[0]
            case 'instructpix2pix':
                source_latents = self.pipe.vae.encode(batch["source"]).latent_dist.sample()
                pose_embeds = self.unet.encode_pose(batch)
                cross_attention_kwargs=dict(source_latents=source_latents, pose_embeds=pose_embeds)
                pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=cross_attention_kwargs)[0]
            case _: raise "unrecognized patch_type"
        
        return noise, latents, noisy_latents, prompt_embeds, multiview_embeds, timesteps, pred

    def compute_loss(self, batch, epoch):
        noise, latents, noisy_latents, prompt_embeds, multiview_embeds, timesteps, pred = self.denoise_one(batch)

        # Compute losses
        losses = {}
        if 'ddim_inv' in self.loss_weights:
            # add the cross attention map to the dictionary
            x_invs = torch.stack([batch['target_inv'][i,t,:,:,:] for i,t in enumerate(timesteps)])
            model_inv_pred = self.unet_inv(x_invs, timesteps, prompt_embeds, return_dict=False)[0]
            target_maps = aggregate_activations(self.unet_inv, detach=True, clear=True)
            source_maps = aggregate_activations(self.unet,     detach=False, clear=True)

            ddim_loss = 0
            for key in target_maps.keys():
                ddim_loss += F.mse_loss(source_maps[key], target_maps[key])
            losses['ddim_inv'] = ddim_loss

        # Get the target for loss depending on the prediction type
        match self.pipe.scheduler.config.prediction_type:
            case "epsilon":      target = noise
            case "v_prediction": target = self.pipe.scheduler.get_velocity(latents, noise, timesteps)
            case _: raise ValueError(f"Unknown prediction type {self.pipe.scheduler.config.prediction_type}")

        losses['denoise'] = F.mse_loss(pred.float(), target.float(), reduction="mean")
        
        return self.aggregate_losses(losses, epoch)

    def aggregate_losses(self, losses, epoch):
        # Aggregate losses
        loss = 0
        #losses = {key:val * self.loss_weights[key] * (self.loss_decays[key] ** epoch) for key, val in losses.items()}
        losses = {key:val * self.loss_weights[key] for key, val in losses.items()}
        for key, val in losses.items():
            loss += val
        losses = {k:v.detach().item() for k,v in losses.items()}
        return loss, {'total': loss.detach().item(), **losses}

    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/run.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/notebooks/explain.ipynb
    def extract_hidden_maps(self, filter_fn, res=16,  clear=True):
        target_maps = aggregate_activations(self.unet_inv, detach=True, clear=clear)
        source_maps = aggregate_activations(self.unet, detach=True, clear=clear)
        target_map = aggregate_attention(target_maps, res=res, filter_fn=filter_fn, cond=True)
        source_maps = aggregate_attention(source_maps, res=res, filter_fn=filter_fn, cond=True)
        return target_map, source_maps

    # see https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    def forward(self, batch, latents=None, guidance_scale=7.5, num_inference_steps=50):
        with torch.inference_mode():
            self.pipe.to(batch['source'].device) 
            assert(batch['target'].shape[0] == 1) # Unsupported batch size for now
            
            multiplier = 2 if guidance_scale > 1 else 1

            match self.patch_type:
                case 'instructpix2pix':
                    source_latents = self.pipe.vae.encode(batch["source"]).latent_dist.sample().repeat(multiplier, 1, 1, 1)
                    pose_embeds = self.unet.encode_pose(batch)
                    pose_embeds = pose_embeds.repeat(multiplier, 1, 1, 1) if pose_embeds is not None else None
                    cross_attention_kwargs=dict(source_latents=source_latents, pose_embeds=pose_embeds)
                case 'crossattention':
                    multiview_embeds = self.multiview_encoder(batch).repeat(multiplier, 1, 1)
                    cross_attention_kwargs=dict(multiview_hidden_states=multiview_embeds)
                case 'transformer_block':
                    multiview_embeds = self.multiview_encoder(batch).repeat(multiplier, 1, 1)
                    cross_attention_kwargs=dict(multiview_hidden_states=multiview_embeds)
                case 'controlnet':
                    raise "Not implemented"
                case _: raise "Unrecognized patch_type"

            input = dict(
                latents=latents, 
                prompt=batch['prompt'], 
                negative_prompt=batch['neg_prompt'], 
                cross_attention_kwargs=cross_attention_kwargs, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )

            self.pipe.unet = self.unet
            pred = self.pipe(**input).images[0]
            
            input = dict(
                latents=batch['target_inv'][:,-1,:,:,:], 
                prompt=batch['prompt'], 
                negative_prompt=batch['neg_prompt'], 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            self.pipe.unet = self.unet_inv
            target = self.pipe(**input).images[0]
            self.pipe.unet = self.unet

        return {'pred': torch.Tensor(np.array(pred)).permute((2,0,1)), 'target': torch.Tensor(np.array(target)).permute((2,0,1))}