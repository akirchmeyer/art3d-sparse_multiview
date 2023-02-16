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

def read_metadata(cls, args, subsample_inst_size):
    meta_df = pd.read_csv(args.meta_data_file, dtype={'index': str})
    inst_ids = {}
    inst_data = []
    meta_df = meta_df.sample(frac = 1) # shuffle
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
        if np.abs(img_data['elev']) <= 0 and np.abs(img_data['azim']) <= 0:
            inst_data[inst_id]['views'].append(img_data)

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

    def __init__(self, args, tokenizer, mode='train'):
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
            raise ValueError(f"Instance {opts.mask_data_path} mask data path doesn't exist.")
        if not Path(opts.inv_data_path).exists():
            raise ValueError(f"Instance {opts.inv_data_path} inv data path doesn't exist.")
        if not Path(opts.prompt_data_path).exists():
            raise ValueError(f"Instance {opts.prompt_data_path} prompt data path doesn't exist.")
            
        # load data
        self.inst_data = read_metadata(opts.cls, opts, self.subsample_inst_size)
        self.num_instances = len(self.inst_data)
        self.num_images = opts.num_images
        self.prompt = opts.prompt
        self.neg_prompt = opts.neg_prompt

        self.data_mode = opts.data_mode
        self.prompt_mode = opts.prompt_mode

        self.diffusion_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size) if self.center_crop else transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = tokenizer
        
        
    def read_img(self, img_path):
        image = Image.open(img_path).convert('RGB')
        return self.diffusion_image_transforms(image)
    
    def imagenet_tranforms(self, numpy_arr):
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

    def __getitem__(self, index):
        inst_id = index % self.num_instances
        inst_data = self.inst_data[inst_id]

        cls = inst_data['class']

        #target_image = Image.open(inst_data['image']).convert("RGB")
        #target_image = torch.Tensor(np.array(target_image)).unsqueeze(0).permute((0,3,1,2))/255

        target_inv = torch.load(inst_data['inv'])
        
        # depth + mask
        mask = torch.Tensor(np.load(inst_data['mask'])).bool()
        
        #center_crop = adjust_image_bbox(mask)
        #target_image[:, :, mask==0] = 1 # mask in white
        #target_image = crop_image(target_image, *center_crop) # center and crop 
        #target_image = F.interpolate(target_image, size=self.size, mode='bilinear').squeeze(0)
        #target_image = 2*target_image-1
        target_image = self.read_img(inst_data['image'])

        # views
        view_id = list(range(len(inst_data['views'])))
        # sample views
        view_id = np.random.choice(view_id, size=1, replace=False)[0] # TODO: replace=True? mask?
        source_data = inst_data['views'][view_id]

        match self.data_mode:
            case 'no_pose':
                source_vp = torch.Tensor([0, 0])
                #source_image = F.interpolate(target_image.unsqueeze(0), size=self.context_size, mode='bilinear')
                source_image = torch.from_numpy(self.imagenet_tranforms(np.load(inst_data['views'][0]['image']))).permute([2,0,1]).float().unsqueeze(0)
            case 'full':
                source_vp = torch.Tensor([source_data['elev'], source_data['azim']])
                source_image = None
            case _:
                raise "Unrecognized data_mode"

        # got OOM errors when using torchvision.transforms
        if source_image is None:
            source_image = np.load(source_data['image'])
            source_image = torch.Tensor(source_image).unsqueeze(0).permute((0,3,1,2))/255 
            mask = source_image[0,:,:,:].mean(axis=0)
            center_crop = adjust_image_bbox(mask < 1)
            source_image = crop_image(source_image, *center_crop) # center and crop 
            source_image = F.interpolate(source_image, size=self.context_size, mode='bilinear')
            source_image = 2*source_image-1

        match self.prompt_mode:
            case 'class': prompt, neg_prompt = inst_data['class'], ''
            case 'data': prompt, neg_prompt = open(inst_data['prompt']).read(), self.neg_prompt
            case 'arg':  prompt, neg_prompt = self.prompt, self.neg_prompt
            
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        # TODO: add positional encoding for viewpoints (MAE)
        return {'target': target_image, 'inv': target_inv, 'source': source_image, 'pose': source_vp, 'class': cls, 'prompt': prompt, 'neg_prompt': neg_prompt, 'input_ids': input_ids }

def resize_image(img, size):
    assert(2 <= len(img.shape) <= 4)
    match len(img.shape):
        case 2: return F.interpolate(img.view(1,1,*img.shape), size=size, mode='bilinear').squeeze()
        case 3: return F.interpolate(img.unsqueeze(0), size=size, mode='bilinear').squeeze(0)
        case 4: return F.interpolate(img, size=size, mode='bilinear')
        case _: raise "Invalid img shape"