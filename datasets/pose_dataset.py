import pickle
import copy
import random

import torch
from torch.utils.data import Dataset
from . import datasets
from . import pipelines as pipes


@datasets.register('PoseDataset')
class PoseDataset(Dataset):
    """
    Dataset for loading pose data.
    """

    def __init__(self, anno_file,
                 pipelines,
                 split=None,
                 test_mode=False,
                 first_k=None,
                 data_ratio=None):
        """
        Args:
            data_path (string): Path to the pose data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.anno_file = anno_file
        self.pipeline = []
        for pipeline in pipelines:
            self.pipeline.append(pipes.make(pipeline))
        self.pipeline = pipes.Compose(self.pipeline)

        self.split = split
        self.test_mode = test_mode
        self.video_infos = self.load_annotations()
        if first_k:
            self.video_infos = self.video_infos[:first_k]
        if data_ratio is not None:
            assert data_ratio > 0 and data_ratio <= 1
            data_len = int(len(self.video_infos) * data_ratio)
            self.video_infos = random.sample(self.video_infos, data_len)
    
    def load_pkl_annotations(self):
        with open(self.anno_file, 'rb') as f:
            data = pickle.load(f)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]
        return data
    
    def load_annotations(self):
        assert self.anno_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['test_mode'] = self.test_mode
        return self.pipeline(results)
    
    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['test_mode'] = self.test_mode
        return self.pipeline(results)

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        return self.prepare_test_frames if self.test_mode else self.prepare_train_frames(idx)