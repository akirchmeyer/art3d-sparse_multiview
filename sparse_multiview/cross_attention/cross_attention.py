import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torchvision import transforms
import numpy as np
import itertools
import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch


from types import SimpleNamespace
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.cross_attention import CrossAttention
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14
import timm
import os
        
from PIL import Image

class CrossAttnProcessorPatch:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs
    ):
        input = hidden_states
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # see https://github.com/pix2pixzero/pix2pix-zero/blob/main/src/utils/cross_attention.py
        #attn.attn_map = attention_probs
        #if not hasattr(attn, 'attn_map') or attn.attn_map is None: 
        #    attn.attn_map = []
        #attn.attn_map.append(attention_probs.detach().cpu())
        #attn.attn_map.append(attention_probs.detach().cpu())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        #if not hasattr(attn, 'hidden_states') or attn.hidden_states is None: 
        #    attn.hidden_states = []
        #attn.hidden_states.append(hidden_states)
        attn.hidden_states = input + hidden_states
        
        return hidden_states

class ResidualCrossAttention(nn.Module):
    def __init__(self, args, text_cross):
        super().__init__()
        self.text_cross = text_cross
        dropout = args.multiview_encoder.residual_ca_dropout
        cross_attention_dim = args.multiview_encoder.cross_attention_dim + (args.multiview_encoder.concat_pose_embed_dim if args.multiview_encoder.pose_embed_mode =='concat' else 0) #TODO: add to params

        self.multiview_cross = CrossAttention(
            query_dim = text_cross.to_q.in_features,
            cross_attention_dim = cross_attention_dim,
            heads = text_cross.heads,
            dim_head = text_cross.to_q.out_features // text_cross.heads,
            dropout = dropout,
            bias = text_cross.to_q.bias,
            upcast_attention = text_cross.upcast_attention,
            upcast_softmax = text_cross.upcast_softmax,
            cross_attention_norm = text_cross.cross_attention_norm, 
            added_kv_proj_dim = None,
            norm_num_groups = None,
            processor = None,
        )
        self.multiview_cross.set_processor(CrossAttnProcessorPatch())

    
    def forward(self, x, encoder_hidden_states=None, attention_mask=None, multiview_hidden_states=None, **cross_attention_kwargs):
        x = self.text_cross(x, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)
        do_classifier_free_guidance = len(encoder_hidden_states) == 2
        return x + self.multiview_cross(x, encoder_hidden_states=torch.cat(2*[multiview_hidden_states]) if do_classifier_free_guidance else multiview_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)


def aggregate_hidden_maps(net, cross_attention_cls = "CrossAttention", cross_attention_layer = 'attn2', step=-1, detach=False, clear=True, map_type='hidden'):
    hidden_maps = {}
    #assert(map_type in ['hidden', 'attention'])
    assert(map_type in ['hidden'])
    
    for name, module in net.named_modules():
        module_name = type(module).__name__
        if module_name == cross_attention_cls and cross_attention_layer in name:
            #hidden_states = module.hidden_states[step] if map_type == 'hidden' else module.attn_map[step] # size is num_channel,s*s,77
            hidden_states = module.hidden_states if map_type == 'hidden' else module.attn_map # size is num_channel,s*s,77
            assert(hidden_states is not None)
            hidden_maps[name] = hidden_states.detach() if detach else hidden_states
            if clear: 
                module.hidden_states = None
                module.attn_map = None
    
    return hidden_maps

def patch_attention_control(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and 'attn2' in name:
            module.set_processor(CrossAttnProcessorPatch())

def patch_residual_attention(args, unet, verbose=True):
    # reset unet crossattention params
    # https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
    def patch_crossattention(model, name, verbose=True):
        for attr in dir(model):
            target_attr = getattr(model, attr)
            if target_attr.__class__.__name__ == 'CrossAttention' and attr == 'attn2':
                if verbose: print(f'replaced: layer={name}.{attr}')
                setattr(model, attr, ResidualCrossAttention(args, target_attr))
            #if type(target_attr) == XFormersCrossAttnProcessor:
            #    target_attr.__call__ = lambda
        for name_layer, layer in model.named_children():
            if layer.__class__.__name__  != 'ResidualCrossAttention':
                patch_crossattention(layer, f'{name}.{name_layer}', verbose)

    patch_crossattention(unet, "unet", verbose=verbose)


def aggregate_attention(attention_maps,
                        res: int,
                        filter_fn,
                        cond = True) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    num_pixels = res ** 2
    idx = 1 if cond else 0 # keep only conditional component
    attention_maps = [v for k, v in attention_maps.items() if filter_fn(k) and v.shape[1] == num_pixels] # (n_maps, 2, num_pixels, num_tokens)
    if len(attention_maps) == 0:
        return []
    attention_maps = torch.stack(attention_maps) # (n_maps, 2, num_pixels, num_tokens)
    attention_maps = attention_maps[:,idx,:,:]   # (n_maps, num_pixels, num_tokens)
    attention_maps = attention_maps.mean(axis=0) # (num_pixels, num_tokens)
    return attention_maps.reshape(res, res, -1).permute(2, 0, 1).cpu() # (num_tokens, res, res)


# from Attend-and-excite vis_utils.py
def show_image_relevance(image_relevance, image, relevnace_res=16, res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    if isinstance(image, Image.Image):
        image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
        image = np.array(image)
    else:
        image = torch.nn.functional.interpolate(image, size=relevnace_res ** 2, mode='bilinear').squeeze().permute((1, 2, 0))
    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    return torch.Tensor(np.array(vis)).resize(res ** 2, res ** 2, 3).permute(2,0,1)
