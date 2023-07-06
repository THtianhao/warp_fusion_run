import sys

import numpy as np
from PIL.ImageDraw import ImageDraw
from safetensors import torch

from scripts.bean.video_bean import VideoConfigBean
from scripts.utils.cmd import gitclone, gitpull
import subprocess
from PIL.Image import Image
from scripts.settings.setting import batchFolder, width_height
from scripts.utils.env import root_dir, root_path
from scripts.utils.ffmpeg_utils import generate_file_hash, extractFrames
from scripts.utils.path import createPath
import os, platform
import shutil
from glob import glob

if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
    print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")

def extra_video_frame(bean: VideoConfigBean):
    if bean.animation_mode == 'Video Input':
        in_path = bean.videoFramesFolder if not bean.flow_video_init_path else bean.flowVideoFramesFolder
        bean.flo_folder = in_path + '_out_flo_fwd'
        bean.temp_flo = in_path + '_temp_flo'
        bean.flo_fwd_folder = in_path + '_out_flo_fwd'
        bean.flo_bck_folder = in_path + '_out_flo_bck'
        postfix = f'{generate_file_hash(bean.video_init_path)[:10]}_{bean.start_frame}_{bean.end_frame_orig}_{bean.extract_nth_frame}'
        if bean.flow_video_init_path:
            flow_postfix = f'{generate_file_hash(bean.flow_video_init_path)[:10]}_{bean.flow_extract_nth_frame}'
        if bean.store_frames_on_google_drive:  # suggested by Chris the Wizard#8082 at discord
            bean.videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
            bean.flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if bean.flow_video_init_path else bean.videoFramesFolder
            bean.condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
            bean.colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
            bean.controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
            bean.recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'

        else:
            bean.videoFramesFolder = f'{root_dir}/videoFrames/{postfix}'
            bean.flowVideoFramesFolder = f'{root_dir}/flowVideoFrames/{flow_postfix}' if bean.flow_video_init_path else bean.videoFramesFolder
            bean.condVideoFramesFolder = f'{root_dir}/condVideoFrames'
            bean.colorVideoFramesFolder = f'{root_dir}/colorVideoFrames'
            bean.controlnetDebugFolder = f'{root_dir}/controlnetDebug'
            bean.recNoiseCacheFolder = f'{root_dir}/recNoiseCache'

        os.makedirs(bean.controlnetDebugFolder, exist_ok=True)
        os.makedirs(bean.recNoiseCacheFolder, exist_ok=True)

        extractFrames(bean.video_init_path, bean.videoFramesFolder, bean.extract_nth_frame, bean.start_frame, bean.end_frame)
        if bean.flow_video_init_path:
            print(bean.flow_video_init_path, bean.flowVideoFramesFolder, bean.flow_extract_nth_frame)
            extractFrames(bean.flow_video_init_path, bean.flowVideoFramesFolder, bean.flow_extract_nth_frame, bean.start_frame, bean.end_frame)

        if bean.cond_video_path:
            print(bean.cond_video_path, bean.condVideoFramesFolder, bean.cond_extract_nth_frame)
            extractFrames(bean.cond_video_path, bean.condVideoFramesFolder, bean.cond_extract_nth_frame, bean.start_frame, bean.end_frame)

        if bean.color_video_path:
            try:
                os.makedirs(bean.colorVideoFramesFolder, exist_ok=True)
                Image.open(bean.color_video_path).save(os.path.join(bean.colorVideoFramesFolder, '000001.jpg'))
            except:
                print(bean.color_video_path, bean.colorVideoFramesFolder, bean.color_extract_nth_frame)
                extractFrames(bean.color_video_path, bean.colorVideoFramesFolder, bean.color_extract_nth_frame, bean.start_frame, bean.end_frame)

def mask_video(bean: VideoConfigBean):
    # Generate background mask from your init video or use a video as a mask
    mask_source = 'init_video'  # @param ['init_video','mask_video']
    # Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
    extract_background_mask = False  # @param {'type':'boolean'}
    # Specify path to a mask video for mask_video mode.
    mask_video_path = ''  # @param {'type':'string'}
    if extract_background_mask:
        os.chdir(root_dir)
        subprocess.run('python -m pip -q install av pims')
        gitclone('https://github.com/Sxela/RobustVideoMattingCLI')
        if mask_source == 'init_video':
            videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(videoFramesAlpha)
            cmd = ['python', f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py", '--input_path', f'{bean.videoFramesFolder}', '--output_alpha', f"{root_dir}/alpha.mp4"]
            process = subprocess.Popen(cmd, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
        if mask_source == 'mask_video':
            videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(videoFramesAlpha)
            maskVideoFrames = bean.videoFramesFolder + 'Mask'
            createPath(maskVideoFrames)
            extractFrames(mask_video_path, f"{maskVideoFrames}", bean.extract_nth_frame, bean.start_frame, bean.end_frame)
            cmd = ['python', f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py", '--input_path', f'{maskVideoFrames}', '--output_alpha', f"{root_dir}/alpha.mp4"]
            process = subprocess.Popen(cmd, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            extractFrames(f"{root_dir}/alpha.mp4", f"{videoFramesAlpha}", 1, 0, 999999999)
    else:
        if mask_source == 'init_video':
            videoFramesAlpha = bean.videoFramesFolder
        if mask_source == 'mask_video':
            videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(videoFramesAlpha)
            extractFrames(mask_video_path, f"{videoFramesAlpha}", bean.extract_nth_frame, bean.start_frame, bean.end_frame)
            # extract video

def download_reference_repository(animation_mode, force: bool = False):
    if (os.path.exists(f'{root_dir}/raft')) and force:
        try:
            shutil.rmtree(f'{root_dir}/raft')
        except:
            print('error deleting existing RAFT model')
    if (not (os.path.exists(f'{root_dir}/raft'))) or force:
        os.chdir(root_dir)
        gitclone('https://github.com/Sxela/WarpFusion')
    else:
        os.chdir(root_dir)
        os.chdir('WarpFusion')
        gitpull()
        os.chdir(root_dir)

    try:
        from python_color_transfer.color_transfer import ColorTransfer, Regrain
    except:
        os.chdir(root_dir)
        gitclone('https://github.com/pengbo-learn/python-color-transfer')

    os.chdir(root_dir)
    sys.path.append('./python-color-transfer')

    if animation_mode == 'Video Input':
        os.chdir(root_dir)
        gitclone('https://github.com/Sxela/flow_tools')

    # @title Define color matching and brightness adjustment
    os.chdir(f"{root_dir}/python-color-transfer")
    from python_color_transfer.color_transfer import ColorTransfer, Regrain
    os.chdir(root_path)

    PT = ColorTransfer()
    RG = Regrain()
