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
from diffusers.models.cross_attention import CrossAttention
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14
import timm
        
from ptp_utils import AttentionStore, aggregate_attention, register_attention_control
from PIL import Image

def read_metadata(cls, meta_data_file, img_data_path, depth_data_path, view_data_path, mask_data_path, subsample_inst_size):
    meta_df = pd.read_csv(meta_data_file, dtype={'index': str})
    inst_ids = {}
    inst_data = []
    meta_df = meta_df.sample(frac = 1) # shuffle
    for _, row in meta_df.iterrows():
        index = row['filename'].split('/')[1][:-4] # TODO: fix hack
        inst, view = [int(x) for x in index.split('.')]
        if not (inst in inst_ids):
            if subsample_inst_size < 0 or len(inst_data) < subsample_inst_size: 
                inst_ids[inst] = len(inst_data)
                inst_data.append({
                    'mask': f'{mask_data_path}/{inst}.npy', 
                    'image':f'{img_data_path}/{inst}.png', 
                    'depth':f'{depth_data_path}/{inst}.npy', 
                    'class':cls,
                    'views':[],
                })
            else:
                continue
        inst_id = inst_ids[inst]
        img_data = {
            'image':f'{view_data_path}/{inst}.{view}.npy', 
            'elev':row['elev'], 
            'azim':row['azim']
        }
        # TODO: filtering -20, 20
        #if np.abs(img_data['elev']) <= 30 and np.abs(img_data['azim']) <= 30:
        inst_data[inst_id]['views'].append(img_data)
    return inst_data

print('Keeping only azim and elev in [-20,20]')

def adjust_image_bbox(mask):
    mask = mask.numpy()
    idxs = np.argwhere(mask)
    min_xy, max_xy = idxs.min(axis=0), idxs.max(axis=0)
    ext_xy = max_xy - min_xy
    pad_xy = (ext_xy.max() - ext_xy[0], ext_xy.max() - ext_xy[1])
    return (min_xy, max_xy, pad_xy)

def crop_image(img, min_xy, max_xy, pad_xy):
    return F.pad(img[..., int(min_xy[0]):int(max_xy[0]), int(min_xy[1]):int(max_xy[1])], (pad_xy[1]//2, pad_xy[1]//2, pad_xy[0]//2, pad_xy[0]//2), "constant", 1,)

# See dreambooth train_dreambooth.py
class MultiViewDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, args, mode='train'):
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
            raise ValueError(f"Instance {opts.mask_data_path} meta data file doesn't exist.")

        # load data
        self.inst_data = read_metadata(opts.cls, opts.meta_data_file, opts.img_data_path, opts.depth_data_path, opts.view_data_path, opts.mask_data_path, self.subsample_inst_size)
        self.num_instances = len(self.inst_data)
        self.num_images = opts.num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        inst_id = index % self.num_instances
        inst_data = self.inst_data[inst_id]

        cls = inst_data['class']

        target_image = Image.open(inst_data['image']).convert("RGB")
        target_image = torch.Tensor(np.array(target_image)).unsqueeze(0).permute((0,3,1,2))/255

        # depth + mask
        mask = torch.Tensor(np.load(inst_data['mask'])).bool()
        center_crop = adjust_image_bbox(mask)
        target_image[:, :, mask==0] = 1 # mask in white
        target_image = crop_image(target_image, *center_crop) # center and crop 
        target_image = F.interpolate(target_image, size=self.size, mode='bilinear').squeeze(0)
        target_image = 2*target_image.contiguous()-1

        # views
        view_id = list(range(len(inst_data['views'])))
        # sample views
        view_id = np.random.choice(view_id, size=1, replace=False)[0] # TODO: replace=True? mask?
        source_data = inst_data['views'][view_id]
        source_vp = torch.Tensor([source_data['elev'], source_data['azim']])

        # got OOM errors when using torchvision.transforms
        source_image = torch.Tensor(np.load(source_data['image'])).unsqueeze(0).permute((0,3,1,2))/255 
        mask = source_image[0,:,:,:].mean(axis=0)
        center_crop = adjust_image_bbox(mask < 1)
        source_image = crop_image(source_image, *center_crop) # center and crop 
        source_image = F.interpolate(source_image, size=self.context_size, mode='bilinear')
        source_image = 2*source_image.contiguous()-1

        # TODO: add positional encoding for viewpoints (MAE)
        return {'target': target_image, 'source': source_image, 'pose': source_vp, 'class': cls}

class ResidualCrossAttention(nn.Module):
    def __init__(self, args, text_cross):
        super().__init__()
        self.text_cross = text_cross
        dropout = args.multiview_encoder.dropout
        cross_attention_dim = args.multiview_encoder.cross_attention_dim #TODO: add to params

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

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, multiview_hidden_states=None, **cross_attention_kwargs):
        x = self.text_cross(x, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)
        do_classifier_free_guidance = len(encoder_hidden_states) == 2
        return x + self.multiview_cross(x, encoder_hidden_states=torch.cat(2*[multiview_hidden_states]) if do_classifier_free_guidance else multiview_hidden_states, attention_mask=attention_mask, **cross_attention_kwargs)

# TODO: add fp16
# See facebook MAE 
# See timm ViT
class MultiViewEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        opts = args.multiview_encoder
        self.model_name = opts.model_name
        match opts.model_name:
            case 'mae':
                self.encode_source = self.mae_encode_source
                self.mae_image_mask_ratio = opts.mae.image_mask_ratio
                match opts.model_size:
                    case 'base': self.model = mae_vit_base_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'large': self.model = mae_vit_large_patch16(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case 'huge': self.model = mae_vit_huge_patch14(checkpoint=opts.mae.model_checkpoint_path, norm_pix_loss=False)
                    case _: raise "Unrecognized MAE model size"
            case 'vit':
                self.encode_source = self.vit_encode_source
                match opts.model_size:
                    case 'base': self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
                    case 'large': self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
                    case 'huge': self.model = timm.create_model('vit_huge_patch14_224', pretrained=True)
                    case _: raise "Unrecognized ViT model size"
            case _: raise "Unrecognized model name"

        # TODO: view mask ratio --> need viewpoint PE
        self.view_mask_ratio = opts.view_mask_ratio

    def mae_encode_source(self, context):
        #_, image, _ = self.mae(image, mask_ratio=self.image_mask_ratio)
        context, _, _ = self.model.forward_encoder(context.view(-1, *context.shape[2:]), mask_ratio=self.mae_image_mask_ratio)
        return context

    def vit_encode_source(self, context):
        context = self.model.forward_features(context.view(-1, *context.shape[2:]))
        return context

    def forward(self, input):
        # x: (N, C, H, W), views: (N, V, C, H, W) , V = n_views_per_inst
        source = input['source']
        # TODO: add reconstruction loss from MAE?
        # TODO: MAE only accepts 224x224
        bsz = source.shape[0]
        source = self.encode_source(source) # (bsz, num_views_per_inst, num_patches, hidden)
        source = source.view(bsz, -1, source.shape[-1]) # (bsz, num_views_per_inst*num_patches, hidden)
        return {'target': input['target'], 'source': source, 'class': input['class'], 'pose': input['pose']}

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel

# See diffusers train_dreambooth.py
class MultiViewDiffusionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load scheduler and models
        opts = args.diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(opts.sd_model)
        self.noise_scheduler = DDPMScheduler.from_pretrained(opts.sd_model, subfolder="scheduler")
        self.multiview_encoder = MultiViewEncoder(args)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.train_multiview_encoder = opts.train_multiview_encoder
        self.with_prior_preservation = opts.with_prior_preservation
        self.prior_loss_weight = opts.prior_loss_weight if self.with_prior_preservation else None

        # reset unet crossattention params
        # https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
        def patch_crossattention(model, name, verbose=False):
            for attr in dir(model):
                target_attr = getattr(model, attr)
                if type(target_attr) == CrossAttention and attr == 'attn2':
                    if verbose: print(f'replaced: layer={name} attr={attr}')
                    setattr(model, attr, ResidualCrossAttention(args, target_attr))
                #if type(target_attr) == XFormersCrossAttnProcessor:
                #    target_attr.__call__ = lambda 
            for name, layer in model.named_children():
                if type(layer) != ResidualCrossAttention:
                    patch_crossattention(layer, name, verbose)

        patch_crossattention(self.unet, "unet", verbose=True)

        # https://huggingface.co/docs/diffusers/optimization/fp16
        self.vae.enable_slicing()

    def finetunable_parameters(self):
        return itertools.chain(self.unet.parameters(), self.multiview_encoder.parameters()) if self.train_multiview_encoder else self.unet.parameters()
    
    def freeze_params(self):
        self.vae.requires_grad_(False)

        for name, params in self.unet.named_parameters():
            params.requires_grad = 'transformer_blocks' in name and 'attn2' in name and 'multiview_cross' in name

        if not self.train_multiview_encoder:
            self.multiview_encoder.requires_grad_(False)

    def finetune(self):
        self.unet.train()
        if self.train_multiview_encoder:
            self.multiview_encoder.train()

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        if self.train_multiview_encoder:
            self.multiview_encoder.gradient_checkpointing_enable()

    def enable_xformers_memory_efficient_attention(self):
        self.unet.enable_xformers_memory_efficient_attention()

    def compute_loss(self, batch):
        # Get the text embedding for conditioning
        batch = self.multiview_encoder(batch)
        self.pipe.to(batch['source'].device)

        # Convert images to latent space
        latents = self.vae.encode(batch["target"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        input_ids = self.pipe.tokenizer(
            batch['class'],
            truncation=True,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(batch['source'].device)

        text_states = self.pipe.text_encoder(input_ids)[0]
        model_pred = self.unet(noisy_latents, timesteps, text_states, cross_attention_kwargs=dict(multiview_hidden_states=batch['source'])).sample

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss


    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/run.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py
    # https://github.com/AttendAndExcite/Attend-and-Excite/blob/main/notebooks/explain.ipynb
    def forward_with_crossattention(self, batch, noise=None, res=16):
        controller = AttentionStore()
        cross_att_count = register_attention_control(self.unet, controller)
        result = self.forward(batch, noise)
        attention_maps = aggregate_attention(attention_store=controller, res=res, from_where=("up", "down", "mid"), is_cross=True, select=0)
        cross_att_count2 = register_attention_control(self.unet, controller, unregister=True)
        assert(cross_att_count == cross_att_count2)
        return result, attention_maps

    # see https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    #TODO: fix unconditional guidance
    def forward(self, batch, latents=None):
        with torch.inference_mode():
            batch = self.multiview_encoder(batch)
            self.pipe.to(batch['source'].device) 
            return self.pipe(latents=latents, prompt=batch['class'], cross_attention_kwargs=dict(multiview_hidden_states=batch['source'])).images[0]
