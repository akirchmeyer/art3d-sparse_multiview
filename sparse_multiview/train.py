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
from cross_attention import show_image_relevance
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
    parser.add_argument("--name", type=str, default='0', required=True)
    parser.add_argument("--mode", type=str, default='train', required=True)
    args = parser.parse_args()
    #args.name = f'{Path(args.config).stem}-{args.id}'
    return OmegaConf.merge(vars(args), OmegaConf.load(args.config))

class Trainer():
    def __init__(self, args):
        self.args = args
        self.opts_trainer = args.trainer
        self.opts_optim = args.optimizer
        self.global_step = 0

        self.init_accelerator()
        
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("multiview_sd", config=vars(args), init_kwargs={"wandb":{"name":args.name}})

        assert(not self.args.visualization.hidden_maps or self.args.visualization.samples)
        self.setup_model()
        self.setup_dataset()
        self.setup_optimizer()
        self.setup_scheduler()

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.train_dataloader_bs1, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.train_dataloader_bs1, self.val_dataloader, self.lr_scheduler)

    def init_accelerator(self):
        logging_dir = Path(self.opts_trainer.output_dir, self.args.name, self.opts_trainer.logging_dir)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(f'{logging_dir}/wandb', exist_ok=True)
        os.environ['WANDB_DIR'] = f'{logging_dir}/wandb'

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.opts_trainer.gradient_accumulation_steps,
            mixed_precision=self.opts_trainer.mixed_precision,
            log_with='wandb',
            logging_dir=logging_dir,
        )

        # Multi-GPU: adapt batch size / gradient accumulation steps
        #effective_batch_size = self.opts_trainer.train_batch_size*self.accelerator.num_processes*self.opts_trainer.gradient_accumulation_steps
        #print(f'effective_batch_size: {effective_batch_size}')
        #print(f'total_batch_size: {self.opts_trainer.total_batch_size}')
        #assert(effective_batch_size >= self.opts_trainer.total_batch_size)
        #assert(effective_batch_size % self.opts_trainer.total_batch_size == 0)
        #self.opts_trainer.train_batch_size = (self.opts_trainer.train_batch_size * self.opts_trainer.total_batch_size) // effective_batch_size 
        #print(f'train_batch_size: {self.opts_trainer.train_batch_size}')
        total_batch_size = self.opts_trainer.train_batch_size * self.accelerator.num_processes * self.opts_trainer.gradient_accumulation_steps
        assert(total_batch_size == self.opts_trainer.total_batch_size)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.opts_trainer.seed is not None:
            set_seed(self.opts_trainer.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.opts_trainer.output_dir is not None:
                os.makedirs(os.path.join(self.opts_trainer.output_dir, self.args.name), exist_ok=True)
        
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        # https://huggingface.co/docs/diffusers/optimization/fp16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    def setup_model(self):
        # import correct text encoder class
        self.model = MultiViewDiffusionModel(self.args)
        self.model.eval()
        self.model.finetune(verbose=True) # debugging
        print('unet params:', sum(p.numel() for p in self.model.parameters()))
        print('unet trainable params:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        

        if self.opts_trainer.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.opts_trainer.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        # Check that all trainable models are in full precision
        # low_precision_error_string = (
        #     "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        #     " doing mixed precision training. copy of the weights should still be float32."
        # )

        # if self.model.unet.dtype != torch.float32:
        #     raise ValueError(
        #         f"Unet loaded as datatype {self.model.unet.dtype}. {low_precision_error_string}"
        #     )

        # if self.model.multiview_encoder.dtype != torch.float32:
        #     raise ValueError(
        #         f"Multiview encoder loaded as datatype {self.model.multiview_encoder.dtype}."
        #         f" {low_precision_error_string}"
        #     ) 

    def setup_optimizer(self):
        if self.opts_optim.scale_lr:
            self.opts_optim.learning_rate = (
                self.opts_optim.learning_rate * self.opts_trainer.gradient_accumulation_steps * self.opts_trainer.train_batch_size * self.accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.opts_trainer.use_8bit_adam:
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
        self.optimizer = optimizer_class(
            self.model.finetunable_parameters(),
            lr=self.opts_optim.learning_rate,
            betas=(self.opts_optim.adam_beta1, self.opts_optim.adam_beta2),
            weight_decay=self.opts_optim.adam_weight_decay,
            eps=self.opts_optim.adam_epsilon,
        )

    def setup_dataset(self):
        # Dataset and DataLoaders creation:
        self.train_dataset = MultiViewDataset(self.args, mode='train')
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.opts_trainer.train_batch_size,
            shuffle=True,
            num_workers=self.opts_trainer.dataloader_num_workers,
            drop_last=True,
            pin_memory=True
        )

        self.train_dataloader_bs1 = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.opts_trainer.dataloader_num_workers,
            drop_last=True,
            pin_memory=True
        )

        # TODO: validation dataset
        self.val_dataset = MultiViewDataset(self.args, mode='val')
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.opts_trainer.dataloader_num_workers,
            pin_memory=True
        )

        if self.accelerator.is_main_process:
            batches = [self.train_dataset[i] for i in range(10)]
            self.accelerator.log({f'dataset': [wandb.Image(torch.cat([batch['target'].cpu(), batch['source'].cpu()], axis=-1)) for batch in batches]})
            #tracker = self.accelerator.get_tracker('wandb')

    def setup_scheduler(self):
        self.num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.opts_trainer.gradient_accumulation_steps)
        self.max_train_steps = self.opts_trainer.num_train_epochs * self.num_update_steps_per_epoch
        self.lr_scheduler = get_scheduler(
            self.opts_optim.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.opts_optim.lr_warmup_steps * self.opts_trainer.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.opts_trainer.gradient_accumulation_steps,
            num_cycles=self.opts_optim.lr_num_cycles,
            power=self.opts_optim.lr_power,
        )

    def train_epoch(self, epoch, progress_bar):
        self.model.eval()
        self.model.finetune()
        self.model.check_parameters(verbose=False)

        self.opts_trainer = self.args.trainer

        progress_bar.set_description(f'Train [{epoch}]')
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
            
                loss, losses = self.model.compute_loss(batch, epoch=epoch)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and self.opts_trainer.max_grad_norm != 'None':
                    self.accelerator.clip_grad_norm_(self.model.finetunable_parameters(), self.opts_trainer.max_grad_norm)

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=self.opts_trainer.set_grads_to_none)
                logs = {**losses, "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

        if (epoch+1) % self.opts_trainer.checkpointing_epochs == 0:
            if self.accelerator.is_main_process:
                save_path = os.path.join(self.opts_trainer.output_dir, self.args.name, f"checkpoint-{self.global_step}")
                self.accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    def evaluate(self, epoch, name):
        self.opts_trainer = self.args.trainer
        self.model.eval()

        match name:
            case 'val':
                dataloader = self.val_dataloader
                num_eval = self.args.visualization.num_val_samples
            case 'train':
                dataloader = self.train_dataloader_bs1
                num_eval = self.args.visualization.num_train_samples
            case _: raise "Unrecognized option"

        progress_bar = tqdm(range(num_eval), disable=not self.accelerator.is_local_main_process, position=1)
        progress_bar.set_description(f'Eval [{epoch}]')

        total_loss = 0
        attn_keys = ['up', 'down', 'mid']
        res = 64
        latents = torch.randn((self.args.visualization.num_latents, 4, 64, 64)).cuda()
        output_imgs, text_maps, view_maps, hidden_maps = [[] for latent in latents], [[] for k in attn_keys], [[] for k in attn_keys], []
        
        #sum_losses = {}
        logs = {}
        for step, batch in enumerate(dataloader):
            if step >= num_eval:
                break
            #with self.accelerator.accumulate(self.model):
            #    loss, losses = self.model.compute_loss(batch, epoch=epoch)
            #    losses = {key:self.accelerator.gather(loss) for key, loss in losses.items()}
            #    sum_losses = {key:sum_losses.get(key,0)+loss for key, loss in losses.items()}

            if self.args.visualization.samples:
                target = 255*(batch['target'].detach().cpu()+1)/2
                source = 255*(batch['source'].detach().cpu()+1)/2
                for i, latent in enumerate(latents):
                    output = self.model(batch, latents=latent.unsqueeze(0))
                    #output_imgs[i].append(torch.cat([target.squeeze(0), source.squeeze(0), output['target'], output['pred']], axis=-1))
                    output_imgs[i].append(torch.cat([target.squeeze(0), source.squeeze(0), output['pred']], axis=-1))

        # Do this only once
        # hidden map
        #sum_losses = {f'{key}_{name}':value/num_eval for key, value in sum_losses.keys()}
        #logs.update(**sum_losses)
        if self.args.visualization.samples:
            logs.update({f'sample_{name}': wandb.Image(torch.cat([torch.cat([img for img in imgs], axis=2) for imgs in output_imgs], axis=1))})
        if self.args.visualization.hidden_maps:
            target_hidden_map, source_hidden_map = self.model.extract_hidden_maps(lambda k: True, res=res, clear=True)
            for j in range(len(target_hidden_map)):
                hidden_maps.append(torch.cat([target_hidden_map[j]/target_hidden_map[j].mean(), source_hidden_map[j]/source_hidden_map[j].mean()], axis=-1))
            logs.update({f'hidden_maps_{name}': [wandb.Image(img) for img in hidden_maps]})
        self.accelerator.log(logs, step=self.global_step)

    def loop(self):
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(self.train_dataloader)}")
        logger.info(f"  Num Epochs = {self.opts_trainer.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.opts_trainer.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.opts_trainer.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.opts_trainer.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")

        first_epoch = 0
        # Potentially load in the weights and states from a previous save
        if self.opts_trainer.resume_from_checkpoint:
            if self.opts_trainer.resume_from_checkpoint != "latest":
                path = os.path.basename(self.opts_trainer.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(os.path.join(self.opts_trainer.output_dir, self.args.name))
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.opts_trainer.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.opts_trainer.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.opts_trainer.output_dir, self.args.name, path))
                first_epoch = int(path.split("-")[1]) + 1

        # Only show the progress bar once on each machine.
        train_progress_bar = tqdm(range(first_epoch*self.num_update_steps_per_epoch, self.max_train_steps), disable=not self.accelerator.is_local_main_process, position=0)

        if self.args.mode == 'train':
            for epoch in range(first_epoch, self.opts_trainer.num_train_epochs):
                self.train_epoch(epoch, train_progress_bar)
                if self.accelerator.is_local_main_process:
                    self.evaluate(epoch, 'train')
                    self.evaluate(epoch, 'val')
        if self.args.mode == 'eval' and self.accelerator.is_local_main_process:
            self.evaluate(first_epoch, 'val')

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    trainer.loop()