import os
import platform
import shutil
import subprocess
import sys

from PIL.Image import Image

from scripts.settings.setting import batchFolder, side_x, side_y
from scripts.utils.cmd import gitclone, gitpull, pipi
from scripts.utils.env import root_dir
from scripts.utils.ffmpeg_utils import generate_file_hash, extractFrames
from scripts.utils.path import createPath
from scripts.video_process.video_config import VideoConfig

if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
    print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")

def set_video_path(config: VideoConfig):
    if config.animation_mode == 'Video Input':
        postfix = f'{generate_file_hash(config.video_init_path)[:10]}_{config.start_frame}_{config.end_frame_orig}_{config.extract_nth_frame}'
        if config.flow_video_init_path:
            flow_postfix = f'{generate_file_hash(config.flow_video_init_path)[:10]}_{config.flow_extract_nth_frame}'
        if config.store_frames_on_google_drive:  # suggested by Chris the Wizard#8082 at discord
            config.videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
            config.flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if config.flow_video_init_path else config.videoFramesFolder
            config.condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
            config.colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
            config.controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
            config.recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'

        else:
            config.videoFramesFolder = f'{root_dir}/videoFrames/{postfix}'
            config.flowVideoFramesFolder = f'{root_dir}/flowVideoFrames/{flow_postfix}' if config.flow_video_init_path else config.videoFramesFolder
            config.condVideoFramesFolder = f'{root_dir}/condVideoFrames'
            config.colorVideoFramesFolder = f'{root_dir}/colorVideoFrames'
            config.controlnetDebugFolder = f'{root_dir}/controlnetDebug'
            config.recNoiseCacheFolder = f'{root_dir}/recNoiseCache'

        config.videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
        config.flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if config.flow_video_init_path else config.videoFramesFolder
        config.condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
        config.colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
        config.controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
        config.recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'
        os.makedirs(config.controlnetDebugFolder, exist_ok=True)
        os.makedirs(config.recNoiseCacheFolder, exist_ok=True)
        config.in_path = config.videoFramesFolder if not config.flow_video_init_path else config.flowVideoFramesFolder
        config.flo_folder = config.flo_fwd_folder = config.in_path + f'_out_flo_fwd/{side_x}_{side_y}'
        config.temp_flo = config.in_path + '_temp_flo'
        config.flo_bck_folder = config.in_path + '_out_flo_bck'

def extra_video_frame(config: VideoConfig):
    extractFrames(config.video_init_path, config.videoFramesFolder, config.extract_nth_frame, config.start_frame, config.end_frame)
    if config.flow_video_init_path:
        print(config.flow_video_init_path, config.flowVideoFramesFolder, config.flow_extract_nth_frame)
        extractFrames(config.flow_video_init_path, config.flowVideoFramesFolder, config.flow_extract_nth_frame, config.start_frame, config.end_frame)

    if config.cond_video_path:
        print(config.cond_video_path, config.condVideoFramesFolder, config.cond_extract_nth_frame)
        extractFrames(config.cond_video_path, config.condVideoFramesFolder, config.cond_extract_nth_frame, config.start_frame, config.end_frame)

    if config.color_video_path:
        try:
            os.makedirs(config.colorVideoFramesFolder, exist_ok=True)
            Image.open(config.color_video_path).save(os.path.join(config.colorVideoFramesFolder, '000001.jpg'))
        except:
            print(config.color_video_path, config.colorVideoFramesFolder, config.color_extract_nth_frame)
            extractFrames(config.color_video_path, config.colorVideoFramesFolder, config.color_extract_nth_frame, config.start_frame, config.end_frame)

def extra_background_mask(bean: VideoConfig):
    if bean.extract_background_mask:
        os.chdir(root_dir)
        pipi('av')
        pipi('pims')
        gitclone('https://github.com/Sxela/RobustVideoMattingCLI')
        if bean.mask_source == 'init_video':
            bean.videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(bean.videoFramesAlpha)
            cmd = ['python', f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py", '--input_path', f'{bean.videoFramesFolder}', '--output_alpha', f"{root_dir}/alpha.mp4"]
            process = subprocess.Popen(cmd, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout)
            print(stderr)
            extractFrames(f"{root_dir}/alpha.mp4", f"{bean.videoFramesAlpha}", 1, 0, 999999999)
        if bean.mask_source == 'mask_video':
            bean.videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(bean.videoFramesAlpha)
            maskVideoFrames = bean.videoFramesFolder + 'Mask'
            createPath(maskVideoFrames)
            extractFrames(bean.mask_video_path, f"{maskVideoFrames}", bean.extract_nth_frame, bean.start_frame, bean.end_frame)
            cmd = ['python', f"{root_dir}/RobustVideoMattingCLI/rvm_cli.py", '--input_path', f'{maskVideoFrames}', '--output_alpha', f"{root_dir}/alpha.mp4"]
            process = subprocess.Popen(cmd, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(process)
            extractFrames(f"{root_dir}/alpha.mp4", f"{bean.videoFramesAlpha}", 1, 0, 999999999)
    else:
        if bean.mask_source == 'init_video':
            bean.videoFramesAlpha = bean.videoFramesFolder
        if bean.mask_source == 'mask_video':
            bean.videoFramesAlpha = bean.videoFramesFolder + 'Alpha'
            createPath(bean.videoFramesAlpha)
            extractFrames(bean.mask_video_path, f"{bean.videoFramesAlpha}", bean.extract_nth_frame, bean.start_frame, bean.end_frame)
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
