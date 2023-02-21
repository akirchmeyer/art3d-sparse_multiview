import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers import AutoencoderKL
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14
import timm
import os
from transformers import ConvNextConfig, ConvNextModel, ConvNextImageProcessor
from transformers.models.convnext.modeling_convnext import ConvNextBackbone

# TODO: add fp16
class MultiViewEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        opts = args.multiview_encoder
        self.concat_pose_embed_dim = opts.concat_pose_embed_dim
        self.cross_attention_dim = opts.cross_attention_dim
        self.pose_embed_mode = opts.pose_embed_mode
        self.pose_embed_type = opts.pose_embed_type

        match self.pose_embed_mode:
            case 'concat': self.total_cross_attention_dim = self.cross_attention_dim + self.concat_pose_embed_dim
            case 'sum': self.total_cross_attention_dim = self.cross_attention_dim
            case _: raise "Unrecognized pose embed mode"

        self.setup_model(opts)

    def setup_model(self, opts):
        self.model_name = opts.model_name
        match self.model_name:
            case 'mae':
                self.mae_image_mask_ratio = opts.mae.image_mask_ratio
                match opts.model_size:
                    case 'base': self.model = mae_vit_base_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'large': self.model = mae_vit_large_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'huge': self.model = mae_vit_huge_patch14(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case _: raise "Unrecognized MAE model size"
            case 'vit':
                match opts.model_size:
                    case 'base': self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
                    case 'large': self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
                    case 'huge': self.model = timm.create_model('vit_huge_patch14_224', pretrained=True)
                    case _: raise "Unrecognized ViT model size"
            case 'convnext':
                # See https://huggingface.co/docs/transformers/model_doc/convnext#transformers.ConvNextModel
                self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
                dim = self.cross_attention_dim
                hidden_sizes = [dim//4, dim//2, dim, dim//2]
                depths = [1, 1, 3, 1]
                print(f'hidden_sizes={hidden_sizes}')
                self.model = ConvNextBackbone(ConvNextConfig(num_channels=4, patch_size=1, hidden_sizes=hidden_sizes, drop_path_rate=0.1, out_features=['stage3']))
            case _: 
                raise "Unrecognized model name"

    def finetune(self):
        match self.model_name:
            case 'mae':
                self.model.requires_grad_(True)
                self.model.train()
            case 'vit':
                self.model.requires_grad_(True)
                self.model.train()
            case 'convnext':
                self.vae.requires_grad_(False)
                self.vae.eval()
                self.model.requires_grad_(True)
                self.model.train()
            case _: 
                raise "Unrecognized model_name"
    
    def finetunable_parameters(self):
        match self.model_name:
            case 'mae':
                return self.model.parameters()
            case 'vit':
                return self.model.parameters()
            case 'convnext':
                return self.model.parameters()
            case _: raise "Unrecognized model_name"
    
    def encode_source(self, context):
        match self.model_name:
            case 'mae':
                context, _, _ = self.model.forward_encoder(context.view(-1, *context.shape[-3:]), mask_ratio=self.mae_image_mask_ratio)
            case 'vit':
                context = self.model.forward_features(context.view(-1, *context.shape[-3:]))
            case 'convnext':
                latents = self.vae.encode(context).latent_dist.sample()
                #latents = latents * self.pipe.vae.config.scaling_factor
                context = self.model(pixel_values=latents, return_dict=False)[0]
                context = context.reshape(*context.shape[:2], -1).permute((0, 2, 1))
            case _:
                raise "Unrecognized model_name"
        return context

    def embed_pose(self, pose, dim):
        match self.pose_embed_type:
            case 'freq': 
                # from TARS
                assert(dim % 4 == 0)
                L, shape, last_dim = dim//4, pose.shape, pose.shape[-1]
                freq = torch.arange(L, dtype=torch.float32, device=pose.device) / L
                freq = 1./10000**freq  # (D/2,) # from Attention is all you need
                spectrum = pose[..., None] * freq  # [B,...,2,L]
                pos_embed = torch.cat([spectrum.sin(), spectrum.cos()], dim=-1).view(*shape[:-1], 2*last_dim*L)  # [B,...,4L]
            case _: 
                raise "Unrecognized pose_embed_type"
        return pos_embed

    def encode_viewpoint(self, source, pose):
        match self.pose_embed_mode:
            case 'concat':
                pos_embed = self.embed_pose(pose, self.concat_pose_embed_dim).unsqueeze(1)
                source = torch.cat([source, pos_embed.repeat(1, source.shape[1], 1)], axis=-1)
            case 'sum':
                pos_embed = self.encode_viewpoint(pose, self.cross_attention_dim).unsqueeze(1)
                source = F.layer_norm(source + pos_embed.repeat(1, source.shape[1], 1), normalized_shape=[pos_embed.shape[-1]], eps=1e-6)
            case _:
                raise "Unrecognized pose_embed_mode"
        return source

    def forward(self, input):
        # x: (N, C, H, W), views: (N, V, C, H, W) , V = n_views_per_inst
        source, pose = input['source_imnet'], input['pose']
        source = self.encode_source(source) # (bsz, num_views_per_inst, num_patches, hidden)
        assert(source.shape[-1] == self.cross_attention_dim) # (bsz, num_views_per_inst*num_patches, hidden)
        source = self.encode_viewpoint(source, pose) # (bsz, num_views_per_inst, num_patches, hidden)
        return source
