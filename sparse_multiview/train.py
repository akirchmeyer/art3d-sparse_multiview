#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import itertools
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Optional

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch.nn.parallel import DistributedDataParallel 
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
import numpy as np
from vis_utils import show_image_relevance
from multiview import MultiViewDiffusionModel, MultiViewEncoder, MultiViewDataset
import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default=None, required=True)
    args = parser.parse_args()
    return OmegaConf.load(args.config)

def train_epoch(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, epoch, progress_bar):
    model.eval()
    model.finetune()
    opts_trainer = args.trainer
    total_loss, n_batches = 0, 0

    progress_bar.set_description(f'Train [{epoch}]')
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            loss = model.compute_loss(batch)
            # https://huggingface.co/blog/accelerate-library
            total_loss += accelerator.gather(loss.item())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.finetunable_parameters(), opts_trainer.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=opts_trainer.set_grads_to_none)
        progress_bar.update(1)
        logs = {"train_loss": total_loss / ((step+1)*accelerator.num_processes), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
    
    accelerator.log(logs, step=epoch)

    if (epoch+1) % opts_trainer.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            save_path = os.path.join(opts_trainer.output_dir, f"checkpoint-{epoch}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

def one_in_key(key, arr):
    for v in arr:
        if v in key: return True
    return False

def evaluate(args, accelerator, model, dataloader, epoch, latents):
    opts_trainer = args.trainer
    res = opts_trainer.res
    model.eval()

    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process, position=1)
    progress_bar.set_description(f'Eval [{epoch}]')

    total_loss = 0
    imgs, cross_maps = [[] for latent in latents], []
    for step, batch in enumerate(dataloader):
        if step >= opts_trainer.num_eval:
            break
        with accelerator.accumulate(model):
            loss = model.compute_loss(batch)
            for i, latent in enumerate(latents):
                latent = latent.unsqueeze(0)

                filter_fn = lambda key: 'cross' in key and one_in_key(key, [f'step{k}' for k in [50]]) and one_in_key(key, ['up', 'down', 'mid'])
                img, attention_maps = model.forward_with_crossattention(batch, latents=latent, filter_fn=filter_fn, avg=True)
                #img = model.forward(batch, latents=latent)

                target = batch['target'].cpu()
                source = batch['source'].cpu()
                imgs[i].append(np.array(torch.cat([target, source, img])))

                if step == 0 and i == 0:
                    target = batch['target'].cpu()
                    source = batch['source'].cpu()
                    for j in range(attention_maps.shape[0]):
                        image_target = show_image_relevance(attention_maps[j, :, :].cpu(), target)
                        image_source = show_image_relevance(attention_maps[j, :, :].cpu(), source)
                        image_target = np.array(Image.fromarray(image_target.astype(np.uint8)).resize((res ** 2, res ** 2)))
                        image_source = np.array(Image.fromarray(image_source.astype(np.uint8)).resize((res ** 2, res ** 2)))
                        cross_maps.append(np.concatenate([image_target, image_source], axis=-1))

            total_loss += accelerator.gather(loss.item())
        progress_bar.update(1)
        logs = {
            "eval_loss": total_loss / ((step+1)*accelerator.num_processes), 
        }
        progress_bar.set_postfix(**logs)
    crosses = {f'crossattention_0_0': [wandb.Image(cross) for cross in cross_maps]}
    images = {f'sample_{i}_{j}': [wandb.Image(img) for img in imgs] for j, imgs in enumerate(imgs)}
    logs.update({
        **images,
        **crosses
    })
    accelerator.log(logs, step=epoch)
    

def main(args):
    opts_trainer = args.trainer
    opts_optim = args.optimizer

    logging_dir = Path(opts_trainer.output_dir, opts_trainer.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=opts_trainer.gradient_accumulation_steps,
        mixed_precision=opts_trainer.mixed_precision,
        log_with='wandb',
        logging_dir=logging_dir,
    )

    # Multi-GPU: adapt batch size / gradient accumulation steps
    effective_batch_size = opts_trainer.train_batch_size*accelerator.num_processes*opts_trainer.gradient_accumulation_steps
    print(f'effective_batch_size: {effective_batch_size}')
    print(f'total_batch_size: {opts_trainer.total_batch_size}')
    assert(effective_batch_size >= opts_trainer.total_batch_size)
    assert(effective_batch_size % opts_trainer.total_batch_size == 0)
    opts_trainer.train_batch_size = (opts_trainer.train_batch_size * opts_trainer.total_batch_size) // effective_batch_size 
    print(f'train_batch_size: {opts_trainer.train_batch_size}')
    total_batch_size = opts_trainer.train_batch_size * accelerator.num_processes * opts_trainer.gradient_accumulation_steps
    assert(total_batch_size == opts_trainer.total_batch_size)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if opts_trainer.seed is not None:
        set_seed(opts_trainer.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if opts_trainer.output_dir is not None:
            os.makedirs(opts_trainer.output_dir, exist_ok=True)

    # import correct text encoder class
    model = MultiViewDiffusionModel(args)
    model.freeze_params()
    model.eval()
    model.finetune(verbose=True) # debugging
    model.eval()

    if opts_trainer.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if opts_trainer.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if model.unet.dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {model.unet.dtype}. {low_precision_error_string}"
        )

    # if model.multiview_encoder.dtype != torch.float32:
    #     raise ValueError(
    #         f"Multiview encoder loaded as datatype {model.multiview_encoder.dtype}."
    #         f" {low_precision_error_string}"
    #     )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # https://huggingface.co/docs/diffusers/optimization/fp16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if opts_optim.scale_lr:
        opts_optim.learning_rate = (
            opts_optim.learning_rate * opts_trainer.gradient_accumulation_steps * opts_trainer.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if opts_trainer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        model.finetunable_parameters(),
        lr=opts_optim.learning_rate,
        betas=(opts_optim.adam_beta1, opts_optim.adam_beta2),
        weight_decay=opts_optim.adam_weight_decay,
        eps=opts_optim.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = MultiViewDataset(args, mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts_trainer.train_batch_size,
        shuffle=True,
        num_workers=opts_trainer.dataloader_num_workers,
        #pin_memory=True
    )
    batches = [train_dataset[i] for i in range(5)]

    # TODO: validation dataset
    val_dataset = MultiViewDataset(args, mode='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opts_trainer.dataloader_num_workers,
        pin_memory=True
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / opts_trainer.gradient_accumulation_steps)
    max_train_steps = opts_trainer.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        opts_optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=opts_optim.lr_warmup_steps * opts_trainer.gradient_accumulation_steps,
        num_training_steps=max_train_steps * opts_trainer.gradient_accumulation_steps,
        num_cycles=opts_optim.lr_num_cycles,
        power=opts_optim.lr_power,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("multiview_sd", config=vars(args))
        accelerator.log({f'dataset_target': [wandb.Image(batch['target'].cpu()) for batch in batches]})
        accelerator.log({f'dataset_source': [wandb.Image(batch['source'].cpu()) for batch in batches]})

        tracker = accelerator.get_tracker('wandb')
        #tracker.define_metric("step")
        #tracker.define_metric("train_loss_ep", step_metric="epoch")
        #tracker.define_metric("val_loss_ep", step_metric="epoch")
    # Train!

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {opts_trainer.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {opts_trainer.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opts_trainer.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if opts_trainer.resume_from_checkpoint:
        if opts_trainer.resume_from_checkpoint != "latest":
            path = os.path.basename(opts_trainer.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(opts_trainer.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{opts_trainer.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            opts_trainer.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(opts_trainer.output_dir, path))
            first_epoch = int(path.split("-")[1]) + 1

    # Only show the progress bar once on each machine.
    train_progress_bar = tqdm(range(first_epoch*num_update_steps_per_epoch, max_train_steps), disable=not accelerator.is_local_main_process, position=0)

    latents = torch.randn((opts_trainer.num_latents, 4, 64, 64)).cuda()
    if opts_trainer.mode == 'train':
        for epoch in range(first_epoch, opts_trainer.num_train_epochs):
            train_epoch(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, epoch, train_progress_bar)
            if accelerator.is_local_main_process:
                evaluate(args, accelerator, model, val_dataloader, epoch, latents)
    if opts_trainer.mode == 'eval' and accelerator.is_local_main_process:
        evaluate(args, accelerator, model, val_dataloader, first_epoch, latents)

    # Create the pipeline using using the trained modules and save it.
    # TODO: save
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     pipeline = DiffusionPipeline.from_pretrained(
    #         opts_trainer.sd_model,
    #         unet=accelerator.unwrap_model(unet),
    #         text_encoder=accelerator.unwrap_model(multiview_encoder),
    #         revision=args.revision,
    #     )
    #     pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)