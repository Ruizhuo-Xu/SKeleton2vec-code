import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange

from .ema import EMA
from . import models
from .utils import trunc_normal_


@models.register('Skeleton2Vec')
class Skeleton2Vec(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(self, encoder_spec, ema_spec,
                 average_top_k_layers,
                 normalize_targets=True,
                 **kwargs):
        super(Skeleton2Vec, self).__init__()
        self.encoder = models.make(encoder_spec)
        self.embed_dim = self.encoder.emb_size
        self.temporal_segment_size = self.encoder.embedding.temporal_segment_size

        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.__dict__.update(kwargs)

        self.regression_head = self._build_regression_head()
        # param init
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.ema = models.make(ema_spec, args={'model': self.encoder})  # EMA acts as the teacher
        self.ema_decay = self.ema.decay
        self.ema_end_decay = self.ema.ema_end_decay
        self.ema_anneal_end_step = self.ema.ema_anneal_end_step

    def _build_regression_head(self):
        """
        Construct the regression head consisting of linear and activation layers.

        Each modality might have its own regression block.

        Returns:
            A nn.Module layer or block of layers
        """
        return nn.Linear(self.embed_dim, self.embed_dim)
        """
        if self.modality == 'text':
            return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim))

        if self.modality in ['audio', 'vision']:
            return nn.Linear(self.embed_dim, self.embed_dim)
        """
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.encoder.encoder.layers):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema(self.encoder)

    def forward(self, src, bool_masked_pos=None, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        bool_masked_pos = bool_masked_pos[:, :, ::self.temporal_segment_size, :, :]
        bool_masked_pos = rearrange(bool_masked_pos, 'B M T V 1 -> (B M) (T V) 1')
        # model forward in online mode (student)
        # x = self.encoder(src, mask, **kwargs)['encoder_out']  # fetch the last layer outputs
        x = self.encoder(src, bool_masked_pos=bool_masked_pos, **kwargs)['last_hidden_state']  # fetch the last layer outputs

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(src, bool_masked_pos=None, **kwargs)['hidden_states']  # fetch the last transformer layers outputs
            y = y[-self.average_top_k_layers:]  # take the last k transformer layers

            # Follow the same layer normalization procedure for text and vision
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            if self.normalize_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])
            """
            if self.modality in ['vision', 'text']:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.model.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            # Use instance normalization for audio
            elif self.modality == 'audio':
                y = [F.instance_norm(tl.float()) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.model.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
            """

        mask = bool_masked_pos.flatten().bool()
        x = rearrange(x, 'b n c -> (b n) c')[mask]
        y = rearrange(y, 'b n c -> (b n) c')[mask]

        x = self.regression_head(x)

        return x, y


@models.register('Skeleton2Vec2')
class Skeleton2Vec2(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(self, model_spec, ema_spec,
                 average_top_k_layers,
                 norm_target_per_layer='layer_norm',
                 normalize_targets=True,
                 **kwargs):
        super(Skeleton2Vec2, self).__init__()
        self.auto_encoder = models.make(model_spec)
        # self.embed_dim = self.encoder.emb_size
        # self.temporal_segment_size = self.encoder.embedding.temporal_segment_size

        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        assert norm_target_per_layer in ['layer_norm', 'instance_norm'], \
        f'norm_target_per_layer must be one of [layer_norm, instance_norm], got {norm_target_per_layer}'
        self.norm_target_per_layer = norm_target_per_layer
        self.__dict__.update(kwargs)

        # param init
        self.apply(self._init_weights)
        # self.fix_init_weight()

        self.ema = models.make(ema_spec, args={'model': self.auto_encoder})  # EMA acts as the teacher
        self.ema_decay = self.ema.decay
        self.ema_end_decay = self.ema.ema_end_decay
        self.ema_anneal_end_step = self.ema.ema_anneal_end_step
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        # if self.ema.decay <= 1:
        self.ema(self.auto_encoder)

    def forward(self, src: torch.Tensor, mask_ratio: float = 0.,
                tube_len: int = 6, num_masked_views: int = 1,
                motion: torch.Tensor = None, norm_motion: bool = True, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        # Multi-mask Training (data2vec2.0)
        src_repeat = src.repeat_interleave(num_masked_views, dim=0)
        x, mask = self.auto_encoder(src_repeat, mask_ratio=mask_ratio, tube_len=tube_len)  # fetch the last layer outputs
        mask = mask.flatten().bool()
        for key, value in x.items():
            x[key] = rearrange(value, 'b n c -> (b n) c')[mask]

        losses = {}
        # model forward in offline mode (teacher)
        if self.auto_encoder.using_feat_head:
            with torch.no_grad():
                self.ema.model.eval()
                y = self.ema.model.forward_encoder(src, mask_ratio=0., output_hidden_states=True)['hidden_states']  # fetch the last transformer layers outputs
                y = y[-self.average_top_k_layers:]  # take the last k transformer layers

                if self.norm_target_per_layer == 'layer_norm':
                    y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                    y = sum(y) / len(y)
                elif self.norm_target_per_layer == 'instance_norm':
                    y = [rearrange(tl, 'b n c -> b c n') for tl in y]
                    y = [F.instance_norm(tl.float()) for tl in y]
                    y = [rearrange(tl, 'b c n -> b n c') for tl in y]
                    y = sum(y) / len(y)
                else:
                    raise NotImplementedError
                if self.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            y = y.repeat_interleave(num_masked_views, dim=0)
            y = rearrange(y, 'b n c -> (b n) c')[mask]
            losses['feat'] = F.mse_loss(x['feat'], y)

        if self.auto_encoder.using_motion_head and motion is not None:
            s = self.auto_encoder.temporal_segment_size
            y_motion = rearrange(motion, 'b n m (t s) v c -> (b n m t v) (s c)', s=s)[mask]
            if norm_motion:
                mean = y_motion.mean(dim=-1, keepdim=True)
                var = y_motion.var(dim=-1, keepdim=True)
                y_motion = (y_motion - mean) / ((var + 1e-6) ** 0.5)
            losses['motion'] = F.mse_loss(x['motion'], y_motion)
        return losses


if __name__ == '__main__':
    encoder_spec = {
        'name': 'SkT',
        'args': {
            'depth': 8
        }
    }
    ema_epce = {
        'name': 'ema',
        'args': {
            'ema_decay': 0.9998,
            'ema_end_decay': 0.9999,
            'ema_anneal_end_step': 300000
        }
    }
    model = Skeleton2Vec(encoder_spec, ema_epce, 4)
    src = torch.randn(3, 2, 120, 25, 3) # B N C
    trg = torch.randn(3, 2, 120, 25, 3) # B N C
    mask = torch.randint(0, 2, (6, 750)).bool()
    inp_data = [src, trg, mask]
    # x, y = model(*inp_data)
    # print(x.shape, y.shape)
    summary(model, input_data=inp_data)