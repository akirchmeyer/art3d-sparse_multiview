import torch
import torch.nn as nn
from diffusers.models.attention import CrossAttention

class MultiViewCrossAttention(nn.Module):
    def __init__(self, cross_attention_dim, dropout, text_cross):
        super().__init__()
        self.text_cross = text_cross
        assert(text_cross.to_q.bias is None)
        self.multiview_cross = CrossAttention(
            query_dim = text_cross.to_q.in_features,
            cross_attention_dim = cross_attention_dim,
            heads = text_cross.heads,
            dim_head = text_cross.to_q.out_features // text_cross.heads,
            dropout = dropout,
            bias = text_cross.to_q.bias is not None,
            upcast_attention = text_cross.upcast_attention,
            upcast_softmax = text_cross.upcast_softmax,
            cross_attention_norm = None, #text_cross.cross_attention_norm, 
            added_kv_proj_dim = None,
            norm_num_groups = None,
            processor = None,
        )
    
    #def forward(self, x, encoder_hidden_states=None, attention_mask=None, multiview_hidden_states=None, x_t=None, **cross_attention_kwargs):
    # def forward(self, x, encoder_hidden_states=None, attention_mask=None, multiview_hidden_states=None, **cross_attention_kwargs):
    #     dx = self.multiview_cross(x, encoder_hidden_states=multiview_hidden_states, attention_mask=attention_mask)
    #     return self.text_cross(x+dx, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

    def forward(self, x, encoder_hidden_states=None, attention_mask=None, multiview_hidden_states=None, **cross_attention_kwargs):
        dx = self.text_cross(x, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)
        return dx + self.multiview_cross(x + dx, encoder_hidden_states=multiview_hidden_states, attention_mask=attention_mask)