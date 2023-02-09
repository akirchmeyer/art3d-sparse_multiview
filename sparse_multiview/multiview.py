import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torchvision import transforms
import numpy as np

from diffusers.models.cross_attention import CrossAttention
from mae.models_mae import mae_vit_large_patch16, mae_vit_base_patch16, mae_vit_huge_patch14


from PIL import Image

class MultiViewDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        img_data_path,
        meta_data_file,
        n_views_per_inst=100,
        size=512,
        mae_size=224,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        if not Path(img_data_path).exists():
            raise ValueError(f"Instance {img_data_path} images root doesn't exist.")
        if not Path(meta_data_file).exists():
            raise ValueError(f"Instance {meta_data_file} meta data file doesn't exist.")

        # load data
        meta_df = pd.read_csv(meta_data_file, dtype={'index': str})
        inst_id = {}
        self.inst_data = []
        self.image_id = []
        for _, row in meta_df.iterrows():
            index = row['filename'].split('/')[1][:-4] # TODO: fix hack
            inst, view = [int(x) for x in index.split('.')]
            if not (inst in inst_id): 
                inst_id[inst] = len(self.inst_data)
                self.inst_data.append([])
            self.image_id.append({'inst_id':inst_id[inst], 'view_id':len(self.inst_data[inst_id[inst]])})
            self.inst_data[inst_id[inst]].append({'image':f'{img_data_path}/{inst}.{view}.png', 'elev':row['elev'], 'azim':row['azim']})
            
        self.num_images = len(self.image_id)
        self.num_views_per_inst = n_views_per_inst
        self.num_instances = len(self.inst_data)

        # image transformation
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.context_image_transforms = transforms.Compose(
            [
                transforms.Resize(mae_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(mae_size) if center_crop else transforms.RandomCrop(mae_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        index = index % self.num_images
        inst_id, view_id = self.image_id[index]['inst_id'], self.image_id[index]['view_id']
        inst_data = self.inst_data[inst_id]
        
        views_id = list(range(len(inst_data)))
        views_id.pop(view_id)
        # TODO: replace=True?
        views_data = [inst_data[i] for i in np.random.choice(views_id, size=self.num_views_per_inst, replace=False)] + [inst_data[view_id]]
        
        # load images
        views = [Image.open(view['image']).convert("RGB") for view in views_data]
        views = [self.context_image_transforms(view) for view in views[:-1]] + [self.image_transforms(views[-1])]

        # views
        image = views.pop(-1)
        views = torch.stack(views)

        # viewpoint information
        viewpoints = [torch.Tensor([view['elev'], view['azim']]) for view in views_data]
        viewpoint = viewpoints.pop(-1)
        rel_viewpoint = torch.stack(viewpoints) - viewpoint

        # TODO: pass relative viewpoints or absolute viewpoints for all images?
        # TODO: add positional encoding for viewpoints (MAE)

        # return
        return {'image': image, 'views': views, 'viewpoints': rel_viewpoint}
    
class MultiViewEncoder(nn.Module):
    def __init__(self, image_mask_ratio=0.75, view_mask_ratio=0.75, model_size='large', n_views_per_inst=5, model_checkpoint=None):
        super().__init__()
        # TODO: change MAE model
        assert(model_checkpoint is not None)
        match model_size:
            case 'base': model_cls = mae_vit_base_patch16
            case 'large': model_cls = mae_vit_large_patch16
            case 'huge': model_cls = mae_vit_huge_patch14
            case _: raise "Unrecognized MAE model size"

        self.mae = model_cls(checkpoint=model_checkpoint, norm_pix_loss=False)
        self.image_mask_ratio = image_mask_ratio
        # TODO: view mask ratio --> need viewpoint PE
        self.view_mask_ratio = view_mask_ratio
        n_patches, embed = self.mae.pos_embed.shape[1:]
        self.mae_embed = embed
        #self.proj = nn.Linear(n_views_per_inst*(int((n_patches-1)*(1-image_mask_ratio))+1)*embed, 1024)

    def forward(self, input):
        # x: (N, C, H, W), views: (N, V, C, H, W) , V = n_views_per_inst
        image, context = input['image'], input['views']
        n, v, c, h, w = context.shape

        # TODO: add reconstruction loss from MAE?
        # TODO: bug, MAE only accepts 224x224
        #_, image, _ = self.mae(image, mask_ratio=self.image_mask_ratio)
        context, _, _ = self.mae.forward_encoder(context.view(-1, c, h, w), mask_ratio=self.image_mask_ratio)
        context = context.view(n, -1, self.mae_embed)
        #context = self.proj(context.view(n, -1))
        return {'image': image, 'context': context}

class CustomCrossAttentionUnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.init_weights()

    def get_named_params(self):
        params_list = []
        for name, params in self.model.named_parameters():
            if 'transformer_blocks' in name and ('attn2.to_k' in name or 'attn2.to_v' in name):
                params_list.append((name, params))
        return params_list

    def set_trainable_cross_kv(self):
        for name, params in self.model.named_parameters():
            params.requires_grad = 'transformer_blocks' in name and ('attn2.to_k' in name or 'attn2.to_v' in name)

    # TODO: check init weights
    def init_weights(self):
        @torch.no_grad()
        def reset_params(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.model.apply(reset_params)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        return self.model(noisy_latents, timesteps, encoder_hidden_states)


    # def hook_model(self):
    #     def new_forward(self, x, context=None, mask=None):
    #         h = self.heads
    #         crossattn = False
    #         if context is not None:
    #             crossattn = True
    #         q = self.to_q(x)
    #         context = default(context, x)
    #         k = self.to_k(context)
    #         v = self.to_v(context)

    #         if crossattn:
    #             modifier = torch.ones_like(k)
    #             modifier[:, :1, :] = modifier[:, :1, :]*0. # why not just 0?? grad?
    #             k = modifier*k + (1-modifier)*k.detach()
    #             v = modifier*v + (1-modifier)*v.detach()

    #         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    #         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    #         attn = sim.softmax(dim=-1)

    #         out = einsum('b i j, b j d -> b i d', attn, v)
    #         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    #         return self.to_out(out)

    #     def change_forward(model):
    #         for layer in model.children():
    #             if type(layer) == CrossAttention:
    #                 bound_method = new_forward.__get__(layer, layer.__class__)
    #                 setattr(layer, 'forward', bound_method)
    #             else:
    #                 change_forward(layer)
    #     change_forward(self.model)