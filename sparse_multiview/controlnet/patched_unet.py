import torch
import torch.nn as nn

from .control_blocks import ControlNetUNet2DConditionModel
from .crossattention_blocks import MultiViewCrossAttention
from .transformer_blocks import MultiViewBasicTransformerBlock
from .instructpix2pix_blocks import InstructPix2PixUNet
from diffusers import UNet2DConditionModel

def patch_unet_with_transformer_block(unet, cross_attention_dim, dropout, verbose=True):
    # https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
    def patch_layers(model, name, verbose=True):
        for name_layer, layer in model.named_children():
            if layer.__class__.__name__ == 'BasicTransformerBlock':
                if verbose: print(f'replaced: layer={name}.{name_layer}')
                setattr(model, name_layer, MultiViewBasicTransformerBlock(cross_attention_dim, dropout, layer))
        for name_layer, layer in model.named_children():
            if layer.__class__.__name__  != 'MultiViewBasicTransformerBlock':
                patch_layers(layer, f'{name}.{name_layer}', verbose)
    patch_layers(unet, "unet", verbose=verbose)

def patch_unet_with_crossattention(unet, cross_attention_dim, dropout, verbose=True):
    # https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
    def patch_layers(model, name, verbose=True):
        for attr in dir(model):
            target_attr = getattr(model, attr)
            if target_attr.__class__.__name__ == 'CrossAttention' and attr == 'attn2':
                if verbose: print(f'replaced: layer={name}.{attr}')
                setattr(model, attr, MultiViewCrossAttention(cross_attention_dim, dropout, target_attr))
        for name_layer, layer in model.named_children():
            if layer.__class__.__name__  != 'MultiViewCrossAttention':
                patch_layers(layer, f'{name}.{name_layer}', verbose)
    patch_layers(unet, "unet", verbose=verbose)



def create_unet(opts, sd_model, verbose=True, **kwargs):
    match opts.patch_type:
        case 'transformer_block': 
            opts = opts.unet_transformer_block
            unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder='unet')
            patch_unet_with_transformer_block(unet, dropout=opts.dropout, cross_attention_dim=kwargs['cross_attention_dim'], verbose=verbose)
        case 'crossattention': 
            opts = opts.unet_crossattention
            unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder='unet')
            patch_unet_with_crossattention(unet, dropout=opts.dropout, cross_attention_dim=kwargs['cross_attention_dim'], verbose=verbose)
        case 'controlnet': 
            opts = opts.unet_controlnet
            #low_cpu_mem_usage=False` and `device_map=None -> randomly initialize missing weights
            controlnet = ControlNetUNet2DConditionModel.from_pretrained(sd_model, subfolder='unet', controlnet_hint_channels=opts.controlnet_hint_channels, low_cpu_mem_usage=False, device_map=None)
            unet       = ControlNetUNet2DConditionModel.from_pretrained(sd_model, subfolder='unet', controlnet_hint_channels=None, low_cpu_mem_usage=False, device_map=None)
            unet = (unet, controlnet)
        case 'instructpix2pix':
            opts = opts.unet_instructpix2pix
            unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder='unet')
            unet = InstructPix2PixUNet(opts, unet=unet)
        case _: raise 'Unrecognized patch_type'

    return unet
