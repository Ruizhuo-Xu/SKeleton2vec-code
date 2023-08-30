# This file includes code from the <ViT> project
# Original source code can be found at <https://github.com/FrancescoSaverioZuppichini/ViT>

# Code obtained from <https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/transfomer.md>
# Author: <FrancescoSaverioZuppichini>

# Code modifications:
# <Adapted skeleton data>

import pdb

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchinfo import summary

import models
from .models import register


class JointEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, temporal_segment_size: int = 4,
                 spatio_size: int = 25, temporal_size: int = 120, emb_size: int = 256):
        self.temporal_segment_size = temporal_segment_size
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('B M T V C -> (B M) C T V'),
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=(temporal_segment_size, 1),
                      stride=(temporal_segment_size, 1)),
            # Rearrange('N C T V -> N (T V) C'),
        )
        self.spatio_pos_emb = nn.Parameter(torch.randn(1, emb_size, 1, spatio_size))
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, emb_size,
                                                         temporal_size // temporal_segment_size, 1))
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x = x + self.spatio_pos_emb + self.temporal_pos_emb
        x = rearrange(x, 'N C T V -> N (T V) C')
        return x
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy * self.scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 8, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x, output_hidden_states: bool = True):
        all_hidden_states = (x,) if output_hidden_states else None
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        
        outputs = (x,) + all_hidden_states if output_hidden_states else (x,)
        return outputs


# class TransformerEncoder(nn.Sequential):
#     def __init__(self, depth: int = 8, **kwargs):
#         super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 256, n_classes: int = 120,
                 num_person: int = 2):
        super().__init__(
            Rearrange('(b m) n e -> b m n e', m=num_person),
            Reduce('b m n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))


# @register('SkT')
# class SkT(nn.Sequential):
#     def __init__(self,     
#                 in_channels: int = 3,
#                 temporal_segment_size: int = 4,
#                 spatio_size: int = 25,
#                 temporal_size: int = 120,
#                 emb_size: int = 256,
#                 depth: int = 12,
#                 n_classes: int = 120,
#                 num_person: int = 2,
#                 drop_p: float = 0.,
#                 forward_drop_p: float = 0.,
#                 **kwargs):
#         super().__init__(
#             JointEmbedding(in_channels, temporal_segment_size,
#                            spatio_size, temporal_size, emb_size),
#             TransformerEncoder(depth, emb_size=emb_size, **kwargs),
#             ClassificationHead(emb_size, n_classes, num_person)
#         )
        

@register('SkT')
class SkT(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                temporal_segment_size: int = 4,
                spatio_size: int = 25,
                temporal_size: int = 120,
                emb_size: int = 256,
                depth: int = 12,
                drop_p: float = 0.,
                forward_drop_p: float = 0.,
                **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.embedding = JointEmbedding(in_channels, temporal_segment_size,
                        spatio_size, temporal_size, emb_size)
        self.encoder = TransformerEncoder(depth, emb_size=emb_size,
                                          drop_p=drop_p,
                                          forward_drop_p=forward_drop_p,
                                          **kwargs)
    
    def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
        x = self.embedding(x)
        encoder_ouputs = self.encoder(x, output_hidden_states=output_hidden_states)
        return {
            'last_hidden_state': encoder_ouputs[0],
            'hidden_states': encoder_ouputs[1:] if output_hidden_states else None,
        }

        
@register('SkTForClassification')
class SkTForClassification(nn.Module):
    def __init__(self, model_spec: dict,
                 num_classes: int = 120,
                 num_person: int = 2):
        super().__init__()
        self.model = models.make(model_spec)
        self.emb_size = self.model.emb_size
        self.cls_head = ClassificationHead(self.emb_size, num_classes, num_person)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        hidden_state = out['last_hidden_state']
        cls_head_out = self.cls_head(hidden_state)
        return cls_head_out


if __name__ == '__main__':
    summary(SkTForClassification(SkT()), (1, 2, 120, 25, 3), device='cpu')