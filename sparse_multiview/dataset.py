import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torchvision import transforms
import numpy as np
import os
        
from PIL import Image
from glob import glob
import random

def read_metadata(cls, args, subsample_inst_size, shuffle=False):
    inst_ids = {}
    inst_data = []

    meta_df = pd.concat([pd.read_csv(filename, dtype={'index': str}) for filename in glob(f'{args.meta_data_path}/*.csv')])
    meta_df = meta_df.sample(frac = 1) if shuffle else meta_df # shuffle
    for _, row in meta_df.iterrows():
        index = row['filename'].split('/')[1][:-4] # TODO: fix hack
        inst, view = [int(x) for x in index.split('.')]
        if not (inst in inst_ids):
            if subsample_inst_size < 0 or len(inst_data) < subsample_inst_size: 
                if not os.path.exists(f'{args.inv_data_path}/{inst}.pt'):
                    continue
                inst_ids[inst] = len(inst_data)
                inst_data.append({
                    'mask': f'{args.mask_data_path}/{inst}.npy', 
                    'image':f'{args.img_data_path}/{inst}.png', 
                    'depth':f'{args.depth_data_path}/{inst}.npy', 
                    'inv':f'{args.inv_data_path}/{inst}.pt', 
                    'prompt':f'{args.prompt_data_path}/{inst}.txt', 
                    'class':cls,
                    'views':[],
                })
            else:
                continue
        inst_id = inst_ids[inst]
        img_data = {
            'image':f'{args.view_data_path}/{inst}.{view}.npy', 
            'elev':row['elev'], 
            'azim':row['azim']
        }
        # TODO: filtering -20, 20
        match args.source_data_mode:
            case 'no_pose':
                if np.abs(img_data['elev']) <= 0 and np.abs(img_data['azim']) <= 0:
                    inst_data[inst_id]['views'].append(img_data)
            case '15deg':
                if np.abs(img_data['elev']) <= 15 and np.abs(img_data['azim']) <= 15:
                    inst_data[inst_id]['views'].append(img_data)
            case 'full':
                inst_data[inst_id]['views'].append(img_data)
            case _:
                raise "Unrecognized source_data_mode"

    return inst_data

print('Keeping only azim and elev in [-0,0]')

def adjust_image_bbox(mask):
    mask = mask.numpy()
    idxs = np.argwhere(mask)
    min_xy, max_xy = idxs.min(axis=0), idxs.max(axis=0)
    ext_xy = max_xy - min_xy
    pad_xy = (ext_xy.max() - ext_xy[0], ext_xy.max() - ext_xy[1])
    return (min_xy, max_xy, pad_xy)

def crop_image(img, min_xy, max_xy, pad_xy):
    return F.pad(img[..., int(min_xy[0]):int(max_xy[0]), int(min_xy[1]):int(max_xy[1])], (pad_xy[1]//2, pad_xy[1]//2, pad_xy[0]//2, pad_xy[0]//2), "constant", 1,)

from transformers import AutoTokenizer

# See dreambooth train_dreambooth.py
class MultiViewDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, args, mode='train'):
        opts = args.dataset
        self.subsample_inst_size = opts[f'{mode}_subsample_inst_size']
        self.shuffle = opts[f'{mode}_shuffle']
        self.size = opts.resolution
        self.center_crop = opts.center_crop

        if not Path(opts.img_data_path).exists():
            raise ValueError(f"Instance {opts.img_data_path} images root doesn't exist.")
        if not Path(opts.meta_data_path).exists():
            raise ValueError(f"Instance {opts.meta_data_path} meta data file doesn't exist.")
        if not Path(opts.mask_data_path).exists():
            raise ValueError(f"Instance {opts.mask_data_path} mask data path doesn't exist.")
        if not Path(opts.inv_data_path).exists():
            raise ValueError(f"Instance {opts.inv_data_path} inv data path doesn't exist.")
        if not Path(opts.prompt_data_path).exists():
            raise ValueError(f"Instance {opts.prompt_data_path} prompt data path doesn't exist.")
            
        # load data

        self.inst_data = read_metadata(opts.cls, opts, self.subsample_inst_size, shuffle=self.shuffle)
        self.num_instances = len(self.inst_data)
        self.num_images = opts.num_images
        self.prompt = opts.prompt
        self.neg_prompt = opts.neg_prompt

        self.source_data_mode = opts.source_data_mode
        self.prompt_mode = opts.prompt_mode

        self.diffusion_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    def read_target_image(self, inst_data):
        target_image = Image.open(inst_data['image']).convert('RGB')
        return self.diffusion_image_transforms(target_image)
    
    def imagenet_tranforms(self, numpy_arr):
        # TODO: we're losing float precision
        img = Image.fromarray(np.asarray(numpy_arr))
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        
        img = img.resize((224, 224))
        img = np.asarray(img) / 255.
        
        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std
        return img

    def __len__(self):
        return self.num_images

    # Output as tensors: (n_channels, height, width)
    def read_ddim_inverse(self, inst_data):
        return torch.load(inst_data['inv'])

    def read_mask(self, inst_data):
        return torch.Tensor(np.load(inst_data['mask'])).bool()

    def sample_view(self, inst_data):
        view_id = list(range(len(inst_data['views'])))
        view_id = np.random.choice(view_id, size=1, replace=False) # TODO: replace=True? mask?
        return view_id[0]

    def read_view_data(self, inst_data, view_id):
        return inst_data['views'][view_id]

    def read_source_pose(self,  inst_data, view_id):
        source_data = self.read_view_data(inst_data, view_id)
        return torch.Tensor([source_data['elev'], source_data['azim']])

    def read_source_image(self, inst_data, view_id):
        source_data = self.read_view_data(inst_data, view_id)
        source_image = Image.fromarray(np.load(source_data['image'])).convert('RGB')
        return self.diffusion_image_transforms(source_image)

    def read_source_image_imnet(self, inst_data, view_id):
        source_data = self.read_view_data(inst_data, view_id)
        source_image = Image.fromarray(np.load(source_data['image'])).convert('RGB')
        source_image = torch.from_numpy(self.imagenet_tranforms(source_image)).permute([2,0,1]).float()
        return source_image

    def read_prompt(self, inst_data):
        match self.prompt_mode:
            case 'class': prompt, neg_prompt = inst_data['class'], ''
            case 'data': prompt, neg_prompt = open(inst_data['prompt']).read(), self.neg_prompt
            case 'arg':  prompt, neg_prompt = self.prompt, self.neg_prompt
        return prompt, neg_prompt

    def __getitem__(self, index):
        inst_id = index % self.num_instances
        inst_data = self.inst_data[inst_id]

        cls = inst_data['class']
        target_inv = self.read_ddim_inverse(inst_data)
        mask = self.read_mask(inst_data)
        target_image = self.read_target_image(inst_data)
        prompt, neg_prompt = self.read_prompt(inst_data)

        # views
        view_id = self.sample_view(inst_data)
        source_pose = self.read_source_pose(inst_data, view_id)
        source_image = self.read_source_image(inst_data, view_id)
        source_image_imnet = self.read_source_image_imnet(inst_data, view_id)
            
        # TODO: add positional encoding for viewpoints (MAE)
        return {'target': target_image, 'target_inv': target_inv, 'source': source_image, 'source_imnet': source_image_imnet, 'pose': source_pose, 'prompt': prompt, 'neg_prompt': neg_prompt }

def resize_image(img, size):
    assert(2 <= len(img.shape) <= 4)
    match len(img.shape):
        case 2: return F.interpolate(img.view(1,1,*img.shape), size=size, mode='bilinear').squeeze()
        case 3: return F.interpolate(img.unsqueeze(0), size=size, mode='bilinear').squeeze(0)
        case 4: return F.interpolate(img, size=size, mode='bilinear')
        case _: raise "Invalid img shape"