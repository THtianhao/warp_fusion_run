import time
from types import SimpleNamespace

import pydevd_pycharm

from scripts.model_process.load_sd_k_model import load_sd_and_k_fusion
from scripts.video_process.video_flow import set_video_path

pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)
import gc
import os
from functools import partial
from glob import glob

import PIL
import numpy as np
import torch
from PIL import Image
# @title ### **Create video**
# @markdown Video file will save in the same folder as your images.
from tqdm.notebook import tqdm

from scripts.model_process.model_config import ModelConfig
from scripts.run.run_common_func import printf
from scripts.run.run_func import apply_mask
from scripts.settings.main_config import MainConfig
from scripts.settings.setting import batchFolder, batch_name
from scripts.utils.env import root_dir, outDirPath
from scripts.video_process.color_transfor_func import warp
from scripts.video_process.video_config import VideoConfig

skip_video_for_run_all = False  # @param {type: 'boolean'}
# @markdown ### **Video masking (post-processing)**
# @markdown Use previously generated background mask during video creation
use_background_mask_video = False  # @param {type: 'boolean'}
invert_mask_video = False  # @param {type: 'boolean'}
# @markdown Choose background source: image, color, init video.
background_video = "init_video"  # @param ['image', 'color', 'init_video']
# @markdown Specify the init image path or color depending on your background video source choice.
background_source_video = 'red'  # @param {type: 'string'}
blend_mode = "optical flow"  # @param ['None', 'linear', 'optical flow']
# if (blend_mode == "optical flow") & (animation_mode != 'Video Input Legacy'):
# @markdown ### **Video blending (post-processing)**
#   print('Please enable Video Input mode and generate optical flow maps to use optical flow blend mode')
blend = 0.5  # @param {type: 'number'}
check_consistency = True  # @param {type: 'boolean'}
postfix = ''
missed_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
# bitrate = 10 #@param {'type':'slider', 'min':'5', 'max':'28', 'step':'1'}
failed_frames = []

def try_process_frame(i, func):
    global failed_frames
    try:
        func(i)
    except:
        print('Error processing frame ', i)

        print('retrying 1 time')
        gc.collect()
        torch.cuda.empty_cache()
        try:
            func(i)
        except Exception as e:
            print('Error processing frame ', i, '. Please lower thread number to 1-3.', e)
            failed_frames.append(i)

if use_background_mask_video:
    postfix += '_mask'
# @markdown #### Upscale settings
upscale_ratio = "1"  # @param [1,2,3,4]
upscale_ratio = int(upscale_ratio)
upscale_model = 'realesr-animevideov3'  # @param ['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealESRGAN_x2plus', 'realesr-animevideov3', 'realesr-general-x4v3']

# @markdown #### Multithreading settings

threads = 12  # @param {type:"number"}
threads = max(min(threads, 64), 1)
frames = []

def generate_video(model_config: ModelConfig, video_config: VideoConfig, main_config: MainConfig):
    if upscale_ratio > 1:
        # try:
        #     for key in loaded_controlnets.keys():
        #         loaded_controlnets[key].cpu()
        # except:
        #     pass
        try:
            model_config.sd_model.model.cpu()
            model_config.sd_model.cond_stage_model.cpu()
            model_config.sd_model.cpu()
            model_config.sd_model.first_stage_model.cpu()
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        os.chdir(f'{root_dir}/Real-ESRGAN')
        print(f'Upscaling to x{upscale_ratio}  using {upscale_model}')
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        os.chdir(root_dir)
        # model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        # netscale = 4
        # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']

        up_model_name = upscale_model
        if up_model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif up_model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif up_model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif up_model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            up_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif up_model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            up_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif up_model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            up_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
        model_path = os.path.join('weights', up_model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = root_dir
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        dni_weight = None

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=up_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device='cuda',
        )

    # @markdown ### **Video settings**

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')

    else:
        # import subprocess in case this cell is run without the above cells
        import subprocess

        from multiprocessing.pool import ThreadPool as Pool

        pool = Pool(threads)
        # 第几批图片？
        latest_run = main_config.args.batchNum

        folder = main_config.args.batch_name  # @param
        run = latest_run  # @param
        final_frame = 'final_frame'

        init_frame = 1  # @param {type:"number"} This is the frame where the video will start
        last_frame = final_frame  # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
        fps = 30  # @param {type:"number"}
        output_format = 'mp4'  # @param ['mp4','mov']
        # view_video_in_cell = True #@param {type: 'boolean'}

        # tqdm.write('Generating video...')

        if last_frame == 'final_frame':
            last_frame = len(glob(batchFolder + f"/{folder}({run})_*.png"))
            print(f'Total frames: {last_frame}')

        image_path = f"{outDirPath}/{folder}/{folder}({run})_%06d.png"
        # filepath = f"{outDirPath}/{folder}/{folder}({run}).{output_format}"
        postfix = ''
        if (blend_mode == 'optical flow') & (True):
            image_path = f"{outDirPath}/{folder}/flow/{folder}({run})_%06d.png"
            postfix += '_flow'
            if upscale_ratio > 1:
                postfix = postfix + f'_x{upscale_ratio}_{upscale_model}'
            video_out = batchFolder + f"/video"
            os.makedirs(video_out, exist_ok=True)
            filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
            if last_frame == 'final_frame':
                last_frame = len(glob(batchFolder + f"/flow/{folder}({run})_*.png"))
            flo_out = batchFolder + f"/flow"
            # !rm -rf {flo_out}/*

            # !mkdir "{flo_out}"
            os.makedirs(flo_out, exist_ok=True)

            frames_in = sorted(glob(batchFolder + f"/{folder}({run})_*.png"))

            frame0 = Image.open(frames_in[0])
            if use_background_mask_video:
                frame0 = apply_mask(frame0, 0, background_video, background_source_video, main_config, video_config, model_config.sd_model, invert_mask_video)
            if upscale_ratio > 1:
                frame0 = np.array(frame0)[..., ::-1]
                output, _ = upsampler.enhance(frame0, outscale=upscale_ratio)
                frame0 = PIL.Image.fromarray((output)[..., ::-1].astype('uint8'))
            frame0.save(flo_out + '/' + frames_in[0].replace('\\', '/').split('/')[-1])

            def process_flow_frame(i):
                frame1_path = frames_in[i - 1]
                frame2_path = frames_in[i]

                frame1 = Image.open(frame1_path)
                frame2 = Image.open(frame2_path)
                frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4]) + 1):06}.jpg"
                flo_path = f"{video_config.flo_folder}/{frame1_stem}.npy"
                weights_path = None
                if check_consistency:
                    if video_config.reverse_cc_order:
                        weights_path = f"{video_config.flo_folder}/{frame1_stem}-21_cc.jpg"
                    else:
                        weights_path = f"{video_config.flo_folder}/{frame1_stem}_12-21_cc.jpg"
                tic = time.time()
                printf('process_flow_frame warp')
                frame = warp(main_config, video_config, frame1, frame2, flo_path, blend=blend, weights_path=weights_path,
                             pad_pct=main_config.padding_ratio, padding_mode=main_config.padding_mode, inpaint_blend=0, video_mode=True)
                if use_background_mask_video:
                    frame = apply_mask(frame, i, background_video, background_source_video, main_config, video_config, model_config.sd_model, invert_mask_video)
                if upscale_ratio > 1:
                    frame = np.array(frame)[..., ::-1]
                    output, _ = upsampler.enhance(frame.clip(0, 255), outscale=upscale_ratio)
                    frame = PIL.Image.fromarray((output)[..., ::-1].clip(0, 255).astype('uint8'))
                frame.save(batchFolder + f"/flow/{folder}({run})_{i:06}.png")

            # process_flow_frame(141)
            with Pool(threads) as p:
                fn = partial(try_process_frame, func=process_flow_frame)
                total_frames = range(init_frame, min(len(frames_in), last_frame))
                result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))

        if blend_mode == 'linear':
            image_path = f"{outDirPath}/{folder}/blend/{folder}({run})_%06d.png"
            postfix += '_blend'
            if upscale_ratio > 1:
                postfix = postfix + f'_x{upscale_ratio}_{upscale_model}'
            video_out = batchFolder + f"/video"
            os.makedirs(video_out, exist_ok=True)
            filepath = f"{video_out}/{folder}({run})_{postfix}.{output_format}"
            if last_frame == 'final_frame':
                last_frame = len(glob(batchFolder + f"/blend/{folder}({run})_*.png"))
            blend_out = batchFolder + f"/blend"
            os.makedirs(blend_out, exist_ok=True)
            frames_in = glob(batchFolder + f"/{folder}({run})_*.png")

            frame0 = Image.open(frames_in[0])
            if use_background_mask_video:
                frame0 = apply_mask(frame0, 0, background_video, background_source_video, main_config, video_config, model_config.sd_model, invert_mask_video)
            if upscale_ratio > 1:
                frame0 = np.array(frame0)[..., ::-1]
                output, _ = upsampler.enhance(frame0.clip(0, 255), outscale=upscale_ratio)
                frame0 = PIL.Image.fromarray((output)[..., ::-1].clip(0, 255).astype('uint8'))
            frame0.save(flo_out + '/' + frames_in[0].replace('\\', '/').split('/')[-1])

            def process_blend_frame(i):
                frame1_path = frames_in[i - 1]
                frame2_path = frames_in[i]

                frame1 = Image.open(frame1_path)
                frame2 = Image.open(frame2_path)
                frame = Image.fromarray((np.array(frame1) * (1 - blend) + np.array(frame2) * (blend)).round().astype('uint8'))
                if use_background_mask_video:
                    frame = apply_mask(frame, i, background_video, background_source_video, main_config, video_config, model_config.sd_model, invert_mask_video)
                if upscale_ratio > 1:
                    frame = np.array(frame)[..., ::-1]
                    output, _ = upsampler.enhance(frame.clip(0, 255), outscale=upscale_ratio)
                    frame = PIL.Image.fromarray((output)[..., ::-1].clip(0, 255).astype('uint8'))
                frame.save(batchFolder + f"/blend/{folder}({run})_{i:06}.png")

            with Pool(threads) as p:
                fn = partial(try_process_frame, func=process_blend_frame)
                total_frames = range(init_frame, min(len(frames_in), last_frame))
                result = list(tqdm(p.imap(fn, total_frames), total=len(total_frames)))
        if output_format == 'mp4':
            cmd = [
                'ffmpeg',
                '-y',
                '-vcodec',
                'png',
                '-r',
                str(fps),
                '-start_number',
                str(init_frame),
                '-i',
                image_path,
                '-frames:v',
                str(last_frame + 1),
                '-c:v',
                'libx264',
                '-vf',
                f'fps={fps}',
                '-pix_fmt',
                'yuv420p',
                # '-crf',
                # f'{bitrate}',
                '-preset',
                'veryslow',
                filepath
            ]
            cmd = [
                'ffmpeg',
                '-y',
                '-vcodec',
                'png',
                '-framerate',
                str(fps),
                '-start_number',
                str(init_frame),
                '-i',
                image_path,
                '-frames:v',
                str(last_frame + 1),
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                filepath
            ]
        if output_format == 'mov':
            cmd = [
                'ffmpeg',
                '-y',
                '-vcodec',
                'png',
                '-r',
                str(fps),
                '-start_number',
                str(init_frame),
                '-i',
                image_path,
                '-frames:v',
                str(last_frame + 1),
                '-c:v',
                'qtrle',
                '-vf',
                f'fps={fps}',
                filepath
            ]

        if upscale_ratio > 1:
            del up_model, upsampler
            gc.collect()
        process = subprocess.Popen(cmd, cwd=f'{batchFolder}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)
        else:
            print(f"The video is ready and saved to {filepath}")
        keep_audio = True  # @param {'type':'boolean'}
        if keep_audio:
            f_audio = filepath[:-4] + '_audio' + filepath[-4:]
            if os.path.exists(filepath) and os.path.exists(video_config.video_init_path):

                cmd_a = ['ffmpeg', '-y', '-i', filepath, '-i', video_config.video_init_path, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-shortest', f_audio]
                process = subprocess.Popen(cmd_a, cwd=f'{root_dir}', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    print(stderr)
                    raise RuntimeError(stderr)
                else:
                    print(f"The video with added audio is saved to {f_audio}")
            else:
                print('Error adding audio from init video to output video: either init or output video don`t exist.')

        # if view_video_in_cell:
        #     mp4 = open(filepath,'rb').read()
        #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        #     display.HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')

if __name__ == "__main__":
    main_config = MainConfig()
    model_config = ModelConfig()
    model_config.force_download = False
    model_config.model_path = '/data/tianhao/stable-diffusion-webui/models/Stable-diffusion/deliberate_v2.safetensors'
    model_config.controlnet_models_dir = '/data/tianhao/warp_fussion/ControlNet/models'
    video_config = VideoConfig()
    video_config.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    set_video_path(video_config)
    load_sd_and_k_fusion(model_config, main_config)
    main_config.args = {
        'batchNum': 0,
        'batch_name': batch_name,

    }
    main_config.args = SimpleNamespace(**main_config.args)
    generate_video(model_config, video_config, main_config)
