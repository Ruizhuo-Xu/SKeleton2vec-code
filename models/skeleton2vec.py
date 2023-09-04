import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from .ema import EMA
from . import models


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
        self.average_top_k_layers = average_top_k_layers
        self.normalize_targets = normalize_targets
        self.__dict__.update(kwargs)

        self.regression_head = self._build_regression_head()
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

    def forward(self, src, trg=None, mask=None, **kwargs):
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
        # model forward in online mode (student)
        # x = self.encoder(src, mask, **kwargs)['encoder_out']  # fetch the last layer outputs
        x = self.encoder(src, **kwargs)['last_hidden_state']  # fetch the last layer outputs
        if trg is None:
            return x

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(trg, **kwargs)['hidden_states']  # fetch the last transformer layers outputs
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

        x = x[mask]
        y = y[mask]

        x = self.regression_head(x)

        return x, y


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