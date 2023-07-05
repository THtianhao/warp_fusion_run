from modules.bean.video_bean import VideoBean
from modules.utils.cmd import gitclone
import subprocess
import sys
import hashlib
import pathlib
from PIL.Image import Image
from glob import glob
from modules.settings.setting import batchFolder
from modules.utils.env import root_dir
from modules.utils.path import createPath
import os, platform

if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
    print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")
def video_setting(bean: VideoBean):
    flow_postfix = ''
    if bean.animation_mode == 'Video Input':
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

def mask_video(bean: VideoBean):
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

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
    createPath(output_path)
    print(f"Exporting Video Frames (1 every {nth_frame})...")
    try:
        for f in [o.replace('\\', '/') for o in glob(output_path + '/*.jpg')]:
            # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
            pathlib.Path(f).unlink()
    except:
        print('error deleting frame ', f)
    # vf = f'select=not(mod(n\\,{nth_frame}))'
    vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
    if os.path.exists(video_path):
        try:
            subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'],
                           stdout=subprocess.PIPE).stdout.decode('utf-8')
        except:
            subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'],
                           stdout=subprocess.PIPE).stdout.decode('utf-8')

    else:
        sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')

def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash

def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash
