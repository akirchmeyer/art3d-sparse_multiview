import os, pdb

import argparse
import numpy as np
import torch
import requests
from PIL import Image
import sys
sys.path.append('pix2pix-zero/src')
sys.path.append('pix2pix-zero/src/utils')
from utils.ddim_inv import DDIMInversion
from utils.scheduler import DDIMInverseScheduler
from types import SimpleNamespace 
import matplotlib.pyplot as plt
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cls', type=str, default='dog')
parser.add_argument('--step', type=int, default=4)
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

cls = args.cls

args = {
    'input_image': f'/grogu/user/akirchme/art3d_sd_webui/fg/{cls}',
    'results_folder': f'/grogu/user/akirchme/art3d_sd_webui/inversion/{cls}',
    'num_ddim_steps': 1000,
    'model_path': 'stabilityai/stable-diffusion-2-1-base',
    'use_float_16': False,
    'prompt': f'{cls}',
    'step': args.step,
    'start': args.start
}

args = SimpleNamespace(**args)

torch_dtype = torch.float16 if args.use_float_16 else torch.float32   
pipe = DDIMInversion.from_pretrained(args.model_path, torch_dtype=torch_dtype).to("cuda")
pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    
def generate_inversion(args):
    # make the output folders
    os.makedirs(args.results_folder, exist_ok=True)

    # if the input is a folder, collect all the images as a list
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
        l_img_paths = list(l_img_paths)[args.start::args.step]
    else:
        l_img_paths = [args.input_image]

    for img_path in l_img_paths:
        bname = os.path.basename(img_path).split(".")[0]
        res_path = os.path.join(args.results_folder, f"{bname}.pt")
        if os.path.exists(res_path):
            continue
        img = Image.open(img_path).resize((512,512), Image.Resampling.LANCZOS)
        res = pipe(
            args.prompt, 
            guidance_scale=1,
            num_inversion_steps=args.num_ddim_steps,
            img=img,
            torch_dtype=torch_dtype,
            return_intermediate=True
        )
        # save the inversion
        torch.save(res, res_path)

generate_inversion(args)