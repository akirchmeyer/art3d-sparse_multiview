import abc

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List
from functools import partial
from diffusers.models.cross_attention import CrossAttention, CrossAttnProcessor

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img




def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img


def register_attention_control(unet, controller, unregister=False):
    def new_processor(
            attn: CrossAttention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            place_in_unet=None,
            **cross_attention_kwargs
        ):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = attn.to_q(hidden_states)

            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.cross_attention_norm:
                encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # store attention matrix to controller
            controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    def register_recr(name, net, count, place_in_unet, unregister=False):
        if net.__class__.__name__ == 'CrossAttention' and 'multiview_cross' in name:
            if unregister:  net.set_processor(CrossAttnProcessor())
            else:           net.set_processor(partial(new_processor, place_in_unet=place_in_unet))
            return count + 1
        elif hasattr(net, 'named_children'):
            for name_, net__ in net.named_children():
                count = register_recr(name_, net__, count, place_in_unet, unregister)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    for name, net in sub_nets:
        if "down" in name:
            cross_att_count += register_recr(name, net, 0, "down", unregister)
        elif "up" in name:
            cross_att_count += register_recr(name, net, 0, "up", unregister)
        elif "mid" in name:
            cross_att_count += register_recr(name, net, 0, "mid", unregister)
    controller.num_att_layers = cross_att_count
    return cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}_step{self.cur_step}"
        if key not in self.attention_store:
            self.attention_store[key] = []
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        self.attention_store[key].append(attn)
        return attn

    def between_steps(self):
        pass

    def get_attention(self):
        average_attention = self.attention_store
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.attention_store = {}

    def __init__(self):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super().__init__()
        self.attention_store = {}
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        filter_fn,
                        is_cross: bool,
                        select: int,
                        avg = True) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_attention()
    num_pixels = res ** 2
    assert(num_pixels <= 32**2) # TODO: see AttentionStore
    for key, item in attention_maps.items():
        item = [it for it in item if it.shape[1] == num_pixels]
        if filter_fn(key) and len(item) > 0:
            item = torch.sum(torch.stack(item), axis=0) / len(item)
            cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    if avg: out = out.sum(0).permute(2, 0, 1) / out.shape[0]
    else: out = out.permute(0,3,1,2).reshape(-1, res, res)
    return out

def visualize_crossattention_map(unet, input, res=16):
    controller = AttentionStore()
    register_attention_control(unet, controller)