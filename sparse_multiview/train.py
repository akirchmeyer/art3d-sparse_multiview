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

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
import numpy as np
from cross_attention.cross_attention import show_image_relevance
from dataset import MultiViewDataset, resize_image
from pipeline import MultiViewDiffusionModel
import wandb
from pathlib import Path
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--id", type=str, default='0', required=True)
    parser.add_argument("--mode", type=str, default='train', required=True)
    args = parser.parse_args()
    args.name = f'{Path(args.config).stem}-{args.id}'
    return OmegaConf.merge(vars(args), OmegaConf.load(args.config))

def train_epoch(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, epoch, global_step, progress_bar):
    model.eval()
    model.finetune()
    model.check_parameters(verbose=False)
    opts_trainer = args.trainer

    progress_bar.set_description(f'Train [{epoch}]')
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            loss, losses = model.compute_loss(batch, epoch=epoch)
            accelerator.backward(loss)
            if accelerator.sync_gradients and opts_trainer.max_grad_norm != 'None':
                accelerator.clip_grad_norm_(model.finetunable_parameters(), opts_trainer.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=opts_trainer.set_grads_to_none)
            progress_bar.update(1)
            logs = {**losses, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step+step)

    if (epoch+1) % opts_trainer.checkpointing_epochs == 0:
        if accelerator.is_main_process:
            save_path = os.path.join(opts_trainer.output_dir, args.name, f"checkpoint-{global_step+step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

def one_in_key(key, arr):
    for v in arr:
        if v in key: return True
    return False

def evaluate(args, accelerator, model, dataloader, epoch, global_step, latents):
    opts_trainer = args.trainer
    model.eval()

    progress_bar = tqdm(range(opts_trainer.num_eval), disable=not accelerator.is_local_main_process, position=1)
    progress_bar.set_description(f'Eval [{epoch}]')

    total_loss = 0
    attn_keys = ['up', 'down', 'mid']
    output_imgs, text_maps, view_maps, hidden_maps = [[] for latent in latents], [[] for k in attn_keys], [[] for k in attn_keys], []
    for step, batch in enumerate(dataloader):
        if step >= opts_trainer.num_eval:
            break

        #with accelerator.accumulate(model):
        #    loss, losses = model.compute_loss(batch, epoch=epoch)
        #    total_loss += accelerator.gather(loss.item())

        for i, latent in enumerate(latents):
            output = model(batch, latents=latent.unsqueeze(0))
            # target, source, output
            target = 255*(batch['target'].detach().cpu()+1)/2
            source = 255*(batch['source'].detach().cpu().squeeze(1)+1)/2
            source = torch.nn.functional.interpolate(source, size=target.shape[-2:], mode='bilinear', align_corners=False)
            output_imgs[i].append(torch.cat([target.squeeze(0), source.squeeze(0), output['target'], output['pred']], axis=-1))

            if step == 0 and i == 0:
                # attention maps
                # for j, key in enumerate(attn_keys):
                #     text_attention_maps, view_attention_maps = model.extract_hidden_maps(lambda k: key in k, res=16, step=-1, clear=False, map_type='attention')
                #     for text_map in text_attention_maps:
                #         text_maps[j].append(show_image_relevance(text_map.cpu(), target, res=16))
                #     for view_map in view_attention_maps:
                #         view_maps[j].append(show_image_relevance(view_map.cpu(), target, res=16))

                # hidden map
                target_hidden_map, source_hidden_map = model.extract_hidden_maps(lambda k: True, res=64, step=-1, clear=True, map_type='hidden')
                #target_hidden_map, source_hidden_map = target_hidden_map.mean(axis=0).unsqueeze(0), source_hidden_map.mean(axis=0).unsqueeze(0)
                for j in range(len(target_hidden_map)):
                    hidden_maps.append(torch.cat([target_hidden_map[j]/target_hidden_map[j].mean(), source_hidden_map[j]/source_hidden_map[j].mean()], axis=-1))

        #progress_bar.update(1)
        #logs = {"eval_loss": total_loss / ((step+1)*accelerator.num_processes)}
        #progress_bar.set_postfix(**logs, **losses)
        
    #text_attention = {f'text_attention_{key}': [wandb.Image(img) for img in text_maps[i]] for i, key in enumerate(attn_keys)} if epoch == 0 else {}
    #view_attention = {f'view_attention_{key}': [wandb.Image(img) for img in view_maps[i]] for i, key in enumerate(attn_keys)}
    hidden = {f'hidden_maps': [wandb.Image(img) for img in hidden_maps]}
    images = {f'sample': wandb.Image(torch.cat([torch.cat([img for img in imgs], axis=2) for imgs in output_imgs], axis=1))}
    logs = {
        **images,
        **hidden,
        #**text_attention,
        #**view_attention
    }
    accelerator.log(logs, step=global_step)
    

def main(args):
    #torch.autograd.set_detect_anomaly(True)


    
    opts_trainer = args.trainer
    opts_optim = args.optimizer

    logging_dir = Path(opts_trainer.output_dir, args.name, opts_trainer.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(f'{logging_dir}/wandb', exist_ok=True)
    os.environ['WANDB_DIR'] = f'{logging_dir}/wandb'

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
            os.makedirs(os.path.join(opts_trainer.output_dir, args.name), exist_ok=True)

    # import correct text encoder class
    model = MultiViewDiffusionModel(args)
    model.freeze_params()
    model.eval()
    model.finetune(verbose=True) # debugging

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
    train_dataset = MultiViewDataset(args, model.tokenizer, mode='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opts_trainer.train_batch_size,
        shuffle=False,
        num_workers=opts_trainer.dataloader_num_workers,
        drop_last=True,
        pin_memory=True
    )
    batches = [train_dataset[i] for i in range(10)]

    # TODO: validation dataset
    val_dataset = MultiViewDataset(args, model.tokenizer, mode='val')
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
        
        accelerator.init_trackers("multiview_sd", config=vars(args), init_kwargs={"wandb":{"name":args.name}})
        accelerator.log({f'dataset': [wandb.Image(torch.cat([batch['target'].cpu(), resize_image(batch['source'][0,:,:,:].cpu(), (512,512))], axis=-1)) for batch in batches]})

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
            dirs = os.listdir(os.path.join(opts_trainer.output_dir, args.name))
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
            accelerator.load_state(os.path.join(opts_trainer.output_dir, args.name, path))
            first_epoch = int(path.split("-")[1]) + 1

    # Only show the progress bar once on each machine.
    train_progress_bar = tqdm(range(first_epoch*num_update_steps_per_epoch, max_train_steps), disable=not accelerator.is_local_main_process, position=0)

    latents = torch.randn((opts_trainer.num_latents, 4, 64, 64)).cuda()
    if args.mode == 'train':
        for epoch in range(first_epoch, opts_trainer.num_train_epochs):
            global_step = epoch*len(train_dataloader)
            train_epoch(args, accelerator, model, optimizer, lr_scheduler, train_dataloader, epoch, global_step, train_progress_bar)
            if accelerator.is_local_main_process:
                evaluate(args, accelerator, model, val_dataloader, epoch, global_step+len(train_dataloader), latents)
    if args.mode == 'eval' and accelerator.is_local_main_process:
        evaluate(args, accelerator, model, val_dataloader, first_epoch, 0, latents)

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