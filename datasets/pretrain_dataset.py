import copy
import pdb

import torch
from torch.utils.data import Dataset
from einops import rearrange

from . import datasets


@datasets.register('PretrainDataset')
class PretrainDataset(Dataset):
    """
    Dataset for pretraining.
    """

    def __init__(self,
                 dataset,
                 mask_ratio: float = 0.6,
                 mask_strategy: str = 'random',
                 temporal_mask_segment_size: int = 4
                ):
        self.dataset = dataset
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        temporal_len = dataset[0]['keypoint'].shape[-3]
        assert temporal_len % temporal_mask_segment_size == 0, \
            'temporal_mask_segment_size should be a divisor of temporal_len'
        self.temporal_mask_segment_size = temporal_mask_segment_size

    def _get_mask(self, input_tensor, mask_ratio, mask_strategy):
        if mask_strategy == 'random':
            # num_clips, num_persons, frames, joints, channels
            N, M, T, V, C = input_tensor.shape
            # get mask martrix, '1' represents mask, '0' represents unmask
            T_ = T // self.temporal_mask_segment_size
            mask = torch.bernoulli(torch.full((N, M, T_, V, 1), mask_ratio))
            mask = torch.repeat_interleave(
                mask, repeats=self.temporal_mask_segment_size, dim=-3)
        else:
            raise NotImplementedError
        return mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data_mask  = copy.deepcopy(data)
        input_tensor = data_mask['keypoint']
        mask = self._get_mask(input_tensor, self.mask_ratio, self.mask_strategy)
        input_tensor = input_tensor * (1 - mask)
        data['keypoint_mask'] = input_tensor
        data['mask'] = mask.bool()

        return data

        
if __name__ == '__main__':
    import yaml
    cfg = 'configs/pretrain_skt.yaml'
    with open(cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    dataset = datasets.make(config['train_dataset']['dataset'])
    dataset = datasets.make(config['train_dataset']['pretrain'], args={'dataset': dataset})
    print(dataset[0].keys())
        
        