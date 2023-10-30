import copy as cp
import numpy as np
import torch
from torch.nn import functional as F
from . import pipelines


@pipelines.register('UniformSampleDecode')
class UniformSampleDecode:

    def __init__(self, clip_len, num_clips=1, p_interval=1, seed=255):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

    # will directly return the decoded clips
    def _get_clips(self, full_kp, clip_len):
        M, T, V, C = full_kp.shape
        clips = []

        for clip_idx in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * T)
            # 随机的起始帧位置
            off = np.random.randint(T - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = (np.arange(start, start + clip_len) % num_frames) + off
                clip = full_kp[:, inds].copy()
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                inds = basic + np.cumsum(offset)[:-1] + off
                clip = full_kp[:, inds].copy()
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset + off
                clip = full_kp[:, inds].copy()
            clips.append(clip)
        return np.concatenate(clips, 1)

    def _handle_dict(self, results):
        assert 'keypoint' in results
        kp = results.pop('keypoint')
        if 'keypoint_score' in results:
            kp_score = results.pop('keypoint_score')
            kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

        kp = kp.astype(np.float32)
        # start_index will not be used
        kp = self._get_clips(kp, self.clip_len)

        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        results['keypoint'] = kp
        return results

    def _handle_list(self, results):
        assert len(results) == self.num_clips
        self.num_clips = 1
        clips = []
        for res in results:
            assert 'keypoint' in res
            kp = res.pop('keypoint')
            if 'keypoint_score' in res:
                kp_score = res.pop('keypoint_score')
                kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

            kp = kp.astype(np.float32)
            kp = self._get_clips(kp, self.clip_len)
            clips.append(kp)
        ret = cp.deepcopy(results[0])
        ret['clip_len'] = self.clip_len
        ret['frame_interval'] = None
        ret['num_clips'] = len(results)
        ret['keypoint'] = np.concatenate(clips, 1)
        self.num_clips = len(results)
        return ret

    def __call__(self, results):
        test_mode = results.get('test_mode', False)
        if test_mode is True:
            np.random.seed(self.seed)
        if isinstance(results, list):
            return self._handle_list(results)
        else:
            return self._handle_dict(results)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'p_interval={self.p_interval}, '
                    f'seed={self.seed})')
        return repr_str

        
@pipelines.register('RandomResizedCrop')
class RandomResizedCrop:
    def __init__(self, p_interval: tuple, clip_len: int, num_clips: int = 1, seed: int = 255):
        self.p_interval = p_interval
        # if not isinstance(p_interval, tuple):
        #     self.p_interval = (p_interval, p_interval)
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed

    def _get_clips(self, full_kp):
        M, T, V, C = full_kp.shape
        begin = 0
        end = T
        valid_size = end - begin
        clips = []
        for clip_idx in range(self.num_clips):
            # crop
            if len(self.p_interval) == 1:
                p = self.p_interval[0]
                bias = int((1 - p) * valid_size / 2)
                clip = full_kp[:, begin+bias:end-bias, :, :]  # center crop
                cropped_length = clip.shape[1]
            else:
                p = np.random.rand(1) * (self.p_interval[1] - self.p_interval[0]) + self.p_interval[0]
                # constraint cropped_length lower bound as 64
                cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)), 64), valid_size)
                bias = np.random.randint(0, valid_size - cropped_length + 1)
                clip = full_kp[:, begin+bias:begin+bias+cropped_length, :, :].copy()
                if clip.shape[1] == 0:
                    print(cropped_length, bias, valid_size)

            # resize
            clip = torch.tensor(clip, dtype=torch.float)
            # clip = clip.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
            clip = clip.permute(3, 2, 0, 1).contiguous().view(C * V * M, cropped_length)
            clip = clip[None, None, :, :]
            # could perform both up sample and down sample
            clip = F.interpolate(clip, size=(C * V * M, self.clip_len), mode='bilinear',align_corners=False).squeeze()
            clip = clip.contiguous().view(C, V, M, self.clip_len).permute(2, 3, 1, 0).contiguous().numpy()
            clips.append(clip)

        return np.concatenate(clips, axis=1)

    def _handle_dict(self, results):
        assert 'keypoint' in results
        kp = results.pop('keypoint')
        if 'keypoint_score' in results:
            kp_score = results.pop('keypoint_score')
            kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

        kp = kp.astype(np.float32)
        # start_index will not be used
        kp = self._get_clips(kp)

        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        results['keypoint'] = kp
        return results

    def _handle_list(self, results):
        assert len(results) == self.num_clips
        self.num_clips = 1
        clips = []
        for res in results:
            assert 'keypoint' in res
            kp = res.pop('keypoint')
            if 'keypoint_score' in res:
                kp_score = res.pop('keypoint_score')
                kp = np.concatenate([kp, kp_score[..., None]], axis=-1)

            kp = kp.astype(np.float32)
            kp = self._get_clips(kp)
            clips.append(kp)
        ret = cp.deepcopy(results[0])
        ret['clip_len'] = self.clip_len
        ret['frame_interval'] = None
        ret['num_clips'] = len(results)
        ret['keypoint'] = np.concatenate(clips, 1)
        self.num_clips = len(results)
        return ret

    def __call__(self, results):
        test_mode = results.get('test_mode', False)
        if test_mode is True:
            np.random.seed(self.seed)
        if isinstance(results, list):
            return self._handle_list(results)
        else:
            return self._handle_dict(results)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'p_interval={self.p_interval}, '
                    f'seed={self.seed})')
        return repr_str

