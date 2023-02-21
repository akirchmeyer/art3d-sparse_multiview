import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import numpy as np

def hook_activations(net, cls = 'BasicTransformerBlock', key='transformer_blocks', attr='norm3', hook_type='output', active=True):
    def extract_activation(name):
        if hook_type == 'output':
            def hook_fn(model, input, output):
                if model.active_hook:
                    model.stored_activation = output
        elif hook_type == 'input':
            def hook_fn(model, input, output):
                if model.active_hook:
                    model.stored_activation = input
        return hook_fn
    
    for name, module in net.named_modules():
        module_name = type(module).__name__
        if module_name == cls and key in name:
            module.active_hook = active
            print(f'Hooked: {name}')
            module.register_forward_hook(extract_activation(attr))

def aggregate_activations(net, cls = 'BasicTransformerBlock', key='transformer_blocks', detach=False, clear=False):
    activations = {}
    for name, module in net.named_modules():
        module_name = type(module).__name__
        if module_name == cls and key in name:
            activations[name] = module.stored_activation if not detach else module.stored_activation.detach()
            if clear: module.stored_activation = None
    return activations

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
