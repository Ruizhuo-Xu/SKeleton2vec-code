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

from . import models
from .drop import DropPath


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
                 att_drop: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(att_drop)
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
                 att_drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 drop_path_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, att_drop=att_drop_p, **kwargs),
                nn.Dropout(forward_drop_p),
                DropPath(drop_path_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(forward_drop_p),
                DropPath(drop_path_p)
            )
            ))


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 8, drop_path_p: float = 0., **kwargs):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_p, depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                drop_path_p=dpr[i], **kwargs)
             for i in range(depth)])

    def forward(self, x, output_hidden_states: bool = True):
        all_hidden_states = (x,) if output_hidden_states else None
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        
        outputs = (x,) + all_hidden_states if output_hidden_states else (x,)
        return outputs
                

@models.register('ClassificationHeadLight')
class ClassificationHeadLight(nn.Sequential):
    def __init__(self, emb_size: int = 256, n_classes: int = 120,
                 num_person: int = 2, drop_p: float = 0.):
        super().__init__(
            nn.LayerNorm(emb_size), 
            Rearrange('(b m) n e -> b m n e', m=num_person),
            Reduce('b m n e -> b e', reduction='mean'),
            nn.Dropout(drop_p),
            nn.Linear(emb_size, n_classes))


@models.register('ClassificationHeadLarge')
class ClassificationHeadLarge(nn.Sequential):
    def __init__(self, emb_size: int = 256,
                 n_classes: int = 120,
                 hidden_dim: int = 2048,
                 num_persons: int = 2,
                 num_joints: int = 25,
                 drop_p: float = 0.):
        super().__init__(
            nn.LayerNorm(emb_size), 
            nn.Dropout(drop_p),
            Rearrange('(B M) (T J) C -> B M T J C', M=num_persons, J=num_joints),
            Reduce('B M T J C -> B M J C', reduction='mean'),
            Rearrange('B M J C -> B M (J C)'),
            Reduce('B M C -> B C', reduction='mean'),
            nn.Linear(emb_size*num_joints, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes)
        )


@models.register('SkT')
class SkT(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                temporal_segment_size: int = 4,
                spatio_size: int = 25,
                temporal_size: int = 120,
                emb_size: int = 256,
                depth: int = 12,
                num_heads: int = 8,
                att_drop_p: float = 0.,
                forward_drop_p: float = 0.,
                drop_path_p: float = 0.,
                **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.embedding = JointEmbedding(in_channels, temporal_segment_size,
                        spatio_size, temporal_size, emb_size)
        self.encoder = TransformerEncoder(depth,
                                          drop_path_p,
                                          emb_size=emb_size,
                                          num_heads=num_heads,
                                          att_drop_p=att_drop_p,
                                          forward_drop_p=forward_drop_p,
                                          **kwargs)
    
    def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
        x = self.embedding(x)
        encoder_ouputs = self.encoder(x, output_hidden_states=output_hidden_states)
        return {
            'last_hidden_state': encoder_ouputs[0],
            'hidden_states': encoder_ouputs[1:] if output_hidden_states else None,
        }

        
@models.register('SkTForClassification')
class SkTForClassification(nn.Module):
    def __init__(self,
                 encoder_spec: dict,
                 cls_head_spec: dict,
                 ):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.emb_size = self.encoder.emb_size

        self.cls_head = models.make(cls_head_spec,
                                    args={'emb_size': self.emb_size})

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)
        hidden_state = out['last_hidden_state']
        cls_head_out = self.cls_head(hidden_state)
        return cls_head_out


if __name__ == '__main__':
    summary(SkTForClassification(SkT()), (1, 2, 120, 25, 3), device='cpu')