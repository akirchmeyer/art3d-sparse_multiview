import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class DepthEstimator(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_types = {
           'high_acc':'DPT_Large',
           'medium_acc':'DPT_Hybrid',
           'low_acc':'MiDaS_small'
        }
        model_type = args['depth_model']
        self.device = args['device']
        self.midas = torch.hub.load("intel-isl/MiDaS", model_types[model_type]).to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        match model_type:
            case 'high_acc':
                transform = midas_transforms.dpt_transform
            case 'medium_acc':
                transform = midas_transforms.dpt_transform
            case 'low_acc':
                transform = midas_transforms.small_transform
            case _:
                raise "Unrecognized depth_model"
                
        self.transform = transform

    # https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/intelisl_midas_v2.ipynb#scrollTo=streaming-assembly
    def forward(self, img):
        """Takes np array as input"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        # imgs: (N, H, W)
        with torch.no_grad():
            input_batch = self.transform(img).to(self.device)
            depth_map = self.midas(input_batch)
            depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False).squeeze()
        return depth_map.cpu().numpy() # TODO: check
    
args = {
    'device': 'cuda',
    'depth_model': 'high_acc',
}
midas = DepthEstimator(args)