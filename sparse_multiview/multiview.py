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
                self.encode_source = self.mae_encode_source
                self.mae_image_mask_ratio = opts.mae.image_mask_ratio

                match opts.model_size:
                    case 'base': self.model = mae_vit_base_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'large': self.model = mae_vit_large_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'huge': self.model = mae_vit_huge_patch14(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case _: raise "Unrecognized MAE model size"
            case 'vit':
                self.encode_source = self.vit_encode_source
                match opts.model_size:
                    case 'base': self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
                    case 'large': self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
                    case 'huge': self.model = timm.create_model('vit_huge_patch14_224', pretrained=True)
                    case _: raise "Unrecognized ViT model size"
            case 'sd':
                self.encode_source = self.sd_encode_source
                self.model = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")

            case _: raise "Unrecognized model name"

        # TODO: view mask ratio --> need viewpoint PE
        self.view_mask_ratio = opts.view_mask_ratio
        self.concat_pose_embed_dim = opts.concat_pose_embed_dim
        self.pose_embed_mode = opts.pose_embed_mode
        self.pose_embed_type = opts.pose_embed_type
        self.cross_attention_dim = opts.cross_attention_dim

        if self.pose_embed_type == 'freq':
            self.encode_viewpoint = self.freq_pose_embed
        else:
            raise "Unrecognized pose_embed_type"

    def mae_encode_source(self, context):
        #_, image, _ = self.mae(image, mask_ratio=self.image_mask_ratio)
        context, mask, ids_restore = self.model.forward_encoder(context.view(-1, *context.shape[2:]), mask_ratio=self.mae_image_mask_ratio)
        return context, (mask, ids_restore)

    def vit_encode_source(self, context):
        context = self.model.forward_features(context.view(-1, *context.shape[2:]))
        return context, None

    def sd_encode_source(self, context):
        context = self.vae.encode(context).latent_dist.sample()
        context = context * self.vae.config.scaling_factor
        # TODO: complete...

    # from tars
    def freq_pose_embed(self, context, L):  # [B,...,2]
        assert(L % 4 == 0)
        L, shape, last_dim = L//4, context.shape, context.shape[-1]
        freq = torch.arange(L, dtype=torch.float32, device=context.device) / L
        freq = 1./10000**freq  # (D/2,) # from Attention is all you need
        spectrum = context[..., None] * freq  # [B,...,2,L]
        points_enc= torch.cat([spectrum.sin(), spectrum.cos()], dim=-1).view(*shape[:-1], 2*last_dim*L)  # [B,...,4L]
        return points_enc

    def forward(self, input):
        # x: (N, C, H, W), views: (N, V, C, H, W) , V = n_views_per_inst
        # TODO: add reconstruction loss from MAE?
        # TODO: MAE only accepts 224x224
        source = input['source']
        source, extra = self.encode_source(source) # (bsz, num_views_per_inst, num_patches, hidden)
        assert(source.shape[-1] == self.cross_attention_dim) # (bsz, num_views_per_inst*num_patches, hidden)
        pose_embed_dim = self.concat_pose_embed_dim if self.pose_embed_mode == 'concat' else self.cross_attention_dim
        pos_embed = self.encode_viewpoint(input['pose'], pose_embed_dim).unsqueeze(1)
        if self.pose_embed_mode == 'concat':
            source = torch.cat([source, pos_embed.repeat(1, source.shape[1], 1)], axis=-1)
        elif self.pose_embed_mode == 'sum':
            source = F.layer_norm(source + pos_embed.repeat(1, source.shape[1], 1), normalized_shape=[pose_embed_dim], eps=1e-6)
        new_input = {'source': source, 'extra': extra}
        return {**input, **new_input}

