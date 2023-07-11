# @title Content-aware scheduing
# @markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
# @markdown rmse function is faster than lpips, but less precise.\
# @markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.
from glob import glob

import numpy as np
import torch
from PIL import Image

from scripts.content_ware_process.content_aware_config import ContentAwareConfig
from scripts.function.poseline import normalize, lpips_model

def load_img_lpips(path, size=(512, 512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    # print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda()

def l1_loss(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))

def rmse(x, y):
    return torch.abs(torch.mean(x - y))

def joint_loss(x, y):
    return rmse(x, y) * lpips_model(x, y)

def content_aware(config: ContentAwareConfig, videoFramesFolder):
    diff_func = rmse
    if config.diff_function == 'lpips':
        diff_func = lpips_model
    if config.diff_function == 'rmse+lpips':
        diff_func = joint_loss

    if config.analyze_video:
        frames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
        from tqdm.notebook import trange

        for i in trange(1, len(frames)):
            with torch.no_grad():
                config.diff.append(diff_func(load_img_lpips(frames[i - 1]), load_img_lpips(frames[i])).sum().mean().detach().cpu().numpy())

        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = [12.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        y = config.diff
        plt.title(f"{config.diff_function} frame difference")
        plt.plot(y, color="red")
        calc_thresh = np.percentile(np.array(config.diff), 97)
        plt.axhline(y=calc_thresh, color='b', linestyle='dashed')

        plt.show()
        print(f'suggested threshold: {calc_thresh.round(2)}')
    # @title Plot threshold vs frame difference
    # @markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
    if config.diff is not None:
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams["figure.figsize"] = [12.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        y = config.diff
        plt.title(f"{config.diff_function} frame difference")
        plt.plot(y, color="red")
        calc_thresh = np.percentile(np.array(config.diff), 97)
        plt.axhline(y=calc_thresh, color='b', linestyle='dashed')
        user_threshold = 0.13  # @param {'type':'raw'}
        plt.axhline(y=user_threshold, color='r')

        plt.show()
        peaks = []
        for i, d in enumerate(config.diff):
            if d > user_threshold:
                peaks.append(i)
        print(f'Peaks at frames: {peaks} for user_threshold of {user_threshold}')
    else:
        print('Please analyze frames in the previous cell  to plot graph')
