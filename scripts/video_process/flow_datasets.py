from glob import glob

import PIL
import numpy as np
import torch
from PIL import Image

from scripts.function.sd_function import normalize
from scripts.settings.setting import width_height, warp_interp
from scripts.video_process.Input_padder import InputPadder

class flowDataset():
    def __init__(self, in_path, half=True, normalize=False):
        frames = sorted(glob(in_path + '/*.*'));
        assert len(frames) > 2, f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.'
        self.frames = frames

    def __len__(self):
        return len(self.frames) - 1

    def load_img(self, img, size):
        img = Image.open(img).convert('RGB').resize(size, warp_interp)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...]

    def __getitem__(self, i):
        frame1, frame2 = self.frames[i], self.frames[i + 1]
        frame1 = self.load_img(frame1, width_height)
        frame2 = self.load_img(frame2, width_height)
        padder = InputPadder(frame1.shape)
        frame1, frame2 = padder.pad(frame1, frame2)
        batch = torch.cat([frame1, frame2])
        if normalize:
            batch = 2 * (batch / 255.0) - 1.0
        return batch
