import copy
import gc
import json
import math
import os
import random
import re
import shutil
import traceback
from datetime import datetime

import PIL
import cv2
import numpy as np
import piexif
import torch
from IPython import display
from PIL import Image, ImageOps
from ipywidgets import Output
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision.transforms.functional as TF

from modules.devices import device
from scripts.captioning_process.generate_key_frame import get_caption
from scripts.clip_process.clip_config import ClipConfig
from scripts.content_ware_process.content_aware_config import ContentAwareConfig
from scripts.lora_embedding.lora_embedding_fun import get_loras_weights_for_frame, load_loras
from scripts.model_process.model_config import ModelConfig
from scripts.refrerence_control_processor.reference_config import ReferenceConfig
from scripts.run.sd_function import match_color_var, run_sd
from scripts.settings.main_config import MainConfig
from scripts.settings.setting import batch_name, batchFolder, side_x, side_y
from scripts.utils.env import root_dir
from scripts.video_process.color_transfor_func import warp, warp_lat, k_means_warp, load_cc, fit, get_flow
from scripts.video_process.video_config import VideoConfig
from scripts.model_process.mode_func import get_image_embed, spherical_dist_loss

stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0 / 200.
VERBOSE = False
blend_json_schedules = False
diffusion_model = "stable_diffusion"
diffusion_sampling_mode = 'ddim'

def do_run(main_config: MainConfig,
           video_config: VideoConfig,
           content_config: ContentAwareConfig,
           model_config: ModelConfig,
           ref_config: ReferenceConfig,
           clip_config: ClipConfig):
    global blend_json_schedules, diffusion_model
    sd_model = model_config.sd_mode
    blend_json_schedules = main_config.blend_json_schedules
    args = main_config.args
    seed = args.seed
    init_image = None
    print(range(args.start_frame, args.max_frames))
    if args.animation_mode != "None":
        batchBar = tqdm(total=args.max_frames, desc="Frames")

    # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
    # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)
    for frame_num in range(args.start_frame, args.max_frames):
        if stop_on_next_loop:
            break

        # display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
            display.display(batchBar.container)
            batchBar.n = frame_num
            batchBar.update(1)
            batchBar.refresh()
            # display.display(batchBar.container)

        # Inits if not video frames
        if args.animation_mode != "Video Input Legacy":
            if args.init_image == '':
                init_image = None
            else:
                init_image = args.init_image
            init_scale = get_scheduled_arg(frame_num, main_config.init_scale_schedule)
            # init_scale = args.init_scale
            steps = int(get_scheduled_arg(frame_num, main_config.steps_schedule))
            style_strength = get_scheduled_arg(frame_num, main_config.style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)
            # skip_steps = args.skip_steps

        if args.animation_mode == 'Video Input':
            if frame_num == args.start_frame:
                steps = int(get_scheduled_arg(frame_num, main_config.steps_schedule))
                style_strength = get_scheduled_arg(frame_num, main_config.style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.skip_steps

                # init_scale = args.init_scale
                init_scale = get_scheduled_arg(frame_num, main_config.init_scale_schedule)
                # init_latent_scale = args.init_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, main_config.latent_scale_schedule)
                init_image = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'
                if main_config.use_background_mask:
                    init_image_pil = Image.open(init_image)
                    init_image_pil = apply_mask(init_image_pil, frame_num, main_config.background, main_config.background_source, main_config.invert_mask)
                    init_image_pil.save(f'init_alpha_{frame_num}.png')
                    init_image = f'init_alpha_{frame_num}.png'
                if (args.init_image != '') and args.init_image is not None:
                    init_image = args.init_image
                    if main_config.use_background_mask:
                        init_image_pil = Image.open(init_image)
                        init_image_pil = apply_mask(init_image_pil, frame_num, main_config.background, main_config.background_source, main_config.invert_mask)
                        init_image_pil.save(f'init_alpha_{frame_num}.png')
                        init_image = f'init_alpha_{frame_num}.png'
                if main_config.VERBOSE: print('init image', args.init_image)
            if frame_num > 0 and frame_num != main_config.frame_range[0]:
                # print(frame_num)

                first_frame_source = batchFolder + f"/{batch_name}({args.batchNum})_{args.start_frame:06}.png"
                if os.path.exists(first_frame_source):
                    first_frame = Image.open(first_frame_source)
                else:
                    first_frame_source = batchFolder + f"/{batch_name}({args.batchNum})_{args.start_frame - 1:06}.png"
                    first_frame = Image.open(first_frame_source)

                # print(frame_num)

                # first_frame = Image.open(batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png")
                # first_frame_source = batchFolder+f"/{batch_name}({batchNum})_{args.start_frame:06}.png"
                if not main_config.fixed_seed:
                    seed += 1
                if main_config.resume_run and frame_num == main_config.args.start_frame:
                    print('if resume_run and frame_num == start_frame')
                    img_filepath = batchFolder + f"/{batch_name}({args.batchNum})_{args.start_frame - 1:06}.png"
                    if args.turbo_mode and frame_num > args.turbo_preroll:
                        shutil.copyfile(img_filepath, 'oldFrameScaled.png')
                    else:
                        shutil.copyfile(img_filepath, 'prevFrame.png')
                else:
                    # img_filepath = '/content/prevFrame.png' if is_colab else 'prevFrame.png'
                    img_filepath = 'prevFrame.png'

                next_step_pil = do_3d_step(img_filepath, frame_num, main_config, video_config, sd_model, forward_clip=args.forward_weights_clip)
                if main_config.warp_mode == 'use_image':
                    next_step_pil.save('prevFrameScaled.png')
                else:
                    # init_image = 'prevFrameScaled_lat.pt'
                    # next_step_pil.save('prevFrameScaled.png')
                    torch.save(next_step_pil, 'prevFrameScaled_lat.pt')

                steps = int(get_scheduled_arg(frame_num, main_config.steps_schedule))
                style_strength = get_scheduled_arg(frame_num, main_config.style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                # skip_steps = args.calc_frames_skip_steps

                ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                if main_config.turbo_mode:
                    if frame_num == main_config.turbo_preroll:  # start tracking oldframe
                        if main_config.warp_mode == 'use_image':
                            next_step_pil.save('oldFrameScaled.png')  # stash for later blending
                        if main_config.warp_mode == 'use_latent':
                            # lat_from_img = get_lat/_from_pil(next_step_pil)
                            torch.save(next_step_pil, 'oldFrameScaled_lat.pt')
                    elif frame_num > main_config.turbo_preroll:
                        # set up 2 warped image sequences, old & new, to blend toward new diff image
                        if main_config.warp_mode == 'use_image':
                            old_frame = do_3d_step('oldFrameScaled.png', frame_num, main_config, video_config, sd_model, forward_clip=main_config.forward_weights_clip_turbo_step)
                            old_frame.save('oldFrameScaled.png')
                        if main_config.warp_mode == 'use_latent':
                            old_frame = do_3d_step('oldFrameScaled.png', frame_num, main_config, video_config, sd_model, forward_clip=main_config.forward_weights_clip_turbo_step)

                            # lat_from_img = get_lat_from_pil(old_frame)
                            torch.save(old_frame, 'oldFrameScaled_lat.pt')
                        if frame_num % int(main_config.turbo_steps) != 0:
                            print('turbo skip this frame: skipping clip diffusion steps')
                            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
                            blend_factor = ((frame_num % int(main_config.turbo_steps)) + 1) / int(main_config.turbo_steps)
                            print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
                            if main_config.warp_mode == 'use_image':
                                newWarpedImg = cv2.imread('prevFrameScaled.png')  # this is already updated..
                                oldWarpedImg = cv2.imread('oldFrameScaled.png')
                                blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0)
                                cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                                next_step_pil.save(f'{img_filepath}')  # save it also as prev_frame to feed next iteration
                            if main_config.warp_mode == 'use_latent':
                                newWarpedImg = torch.load('prevFrameScaled_lat.pt')  # this is already updated..
                                oldWarpedImg = torch.load('oldFrameScaled_lat.pt')
                                blendedImage = newWarpedImg * (blend_factor) + oldWarpedImg * (1 - blend_factor)
                                blendedImage = get_image_from_lat(blendedImage, sd_model).save(f'{batchFolder}/{filename}')
                                torch.save(next_step_pil, f'{img_filepath[:-4]}_lat.pt')

                            if main_config.turbo_frame_skips_steps is not None:
                                if main_config.warp_mode == 'use_image':
                                    oldWarpedImg = cv2.imread('prevFrameScaled.png')
                                    cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
                                print('clip/diff this frame - generate clip diff image')
                                if main_config.warp_mode == 'use_latent':
                                    oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                                    torch.save(
                                        oldWarpedImg,
                                        f'oldFrameScaled_lat.pt',
                                    )  # swap in for blending later
                                skip_steps = math.floor(steps * main_config.turbo_frame_skips_steps)
                            else:
                                continue
                        else:
                            # if not a skip frame, will run diffusion and need to blend.
                            if main_config.warp_mode == 'use_image':
                                oldWarpedImg = cv2.imread('prevFrameScaled.png')
                                cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
                            print('clip/diff this frame - generate clip diff image')
                            if main_config.warp_mode == 'use_latent':
                                oldWarpedImg = torch.load('prevFrameScaled_lat.pt')
                                torch.save(
                                    oldWarpedImg,
                                    f'oldFrameScaled_lat.pt',
                                )  # swap in for blending later
                            # oldWarpedImg = cv2.imread('prevFrameScaled.png')
                            # cv2.imwrite(f'oldFrameScaled.png',oldWarpedImg)#swap in for blending later
                            print('clip/diff this frame - generate clip diff image')
                if main_config.warp_mode == 'use_image':
                    init_image = 'prevFrameScaled.png'
                else:
                    init_image = 'prevFrameScaled_lat.pt'
                if main_config.use_background_mask:
                    if main_config.warp_mode == 'use_latent':
                        # pass
                        latent = apply_mask(latent.cpu(), frame_num, main_config.background, main_config.background_source, main_config.invert_mask, main_config.warp_mode)  # .save(init_image)

                    if main_config.warp_mode == 'use_image':
                        apply_mask(Image.open(init_image), frame_num, main_config.background, main_config.background_source, main_config.invert_mask).save(init_image)
                # init_scale = args.frames_scale
                init_scale = get_scheduled_arg(frame_num, main_config.init_scale_schedule)
                # init_latent_scale = args.frames_latent_scale
                init_latent_scale = get_scheduled_arg(frame_num, main_config.latent_scale_schedule)

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            # frame_prompt = args.prompts_series[-1]
            frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
        elif args.prompts_series is not None:
            # frame_prompt = args.prompts_series[frame_num]
            frame_prompt = get_sched_from_json(frame_num, args.prompts_series, blend=False)
        else:
            frame_prompt = []

        if VERBOSE: print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
        elif args.image_prompts_series is not None:
            image_prompt = get_sched_from_json(frame_num, args.image_prompts_series, blend=False)
        else:
            image_prompt = []

        init = None

        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == 'None':
                display.clear_output(wait=True)
                batchBar = tqdm(range(args.n_batches), desc="Batches")
                batchBar.n = i
                batchBar.refresh()
            print('')
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            steps = int(get_scheduled_arg(frame_num, main_config.steps_schedule))
            style_strength = get_scheduled_arg(frame_num, main_config.style_strength_schedule)
            skip_steps = int(steps - steps * style_strength)

            if main_config.perlin_init:
                init = regen_perlin(main_config.perlin_mode, main_config.batch_size, side_x, side_y)

            consistency_mask = None
            if (main_config.check_consistency or (model_version == 'v1_inpainting') or ('control_sd15_inpaint' in main_config.controlnet_multimodel.keys())) and frame_num > 0:
                frame1_path = f'{video_config.videoFramesFolder}/{frame_num:06}.jpg'
                if video_config.reverse_cc_order:
                    weights_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                else:
                    weights_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"
                consistency_mask = load_cc(main_config, weights_path, blur=main_config.consistency_blur, dilate=main_config.consistency_dilate)

            if diffusion_model == 'stable_diffusion':
                if VERBOSE: print(args.side_x, args.side_y, init_image)
                # init = Image.open(fetch(init_image)).convert('RGB')

                # init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
                # init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
                # text_prompt = copy.copy(args.prompts_series[frame_num])
                text_prompt = copy.copy(get_sched_from_json(frame_num, args.prompts_series, blend=False))
                if VERBOSE: print(f'Frame {frame_num} Prompt: {text_prompt}')
                text_prompt = [re.sub('\<(.*?)\>', '', o).strip(' ') for o in text_prompt]  # remove loras from prompt
                used_loras, used_loras_weights = get_loras_weights_for_frame(frame_num, main_config.new_prompt_loras)
                if VERBOSE:
                    print('used_loras, used_loras_weights', used_loras, used_loras_weights)
                # used_loras_weights = [o for o in used_loras_weights if o is not None else 0.]
                load_loras(used_loras, used_loras_weights)
                caption = get_caption(frame_num)
                if caption:
                    # print('args.prompt_series',args.prompts_series[frame_num])
                    if '{caption}' in text_prompt[0]:
                        print('Replacing ', '{caption}', 'with ', caption)
                        text_prompt[0] = text_prompt[0].replace('{caption}', caption)
                prompt_patterns = get_sched_from_json(frame_num, main_config.prompt_patterns_sched, blend=False)
                if prompt_patterns:
                    for key in prompt_patterns.keys():
                        if key in text_prompt[0]:
                            print('Replacing ', key, 'with ', prompt_patterns[key])
                            text_prompt[0] = text_prompt[0].replace(key, prompt_patterns[key])

                if args.neg_prompts_series is not None:
                    neg_prompt = get_sched_from_json(frame_num, args.neg_prompts_series, blend=False)
                else:
                    neg_prompt = copy.copy(text_prompt)

                if VERBOSE: print(f'Frame {frame_num} neg_prompt: {neg_prompt}')
                if args.rec_prompts_series is not None:
                    rec_prompt = copy.copy(get_sched_from_json(frame_num, args.rec_prompts_series, blend=False))
                    if caption and '{caption}' in rec_prompt[0]:
                        print('Replacing ', '{caption}', 'with ', caption)
                        rec_prompt[0] = rec_prompt[0].replace('{caption}', caption)
                else:
                    rec_prompt = copy.copy(text_prompt)
                if VERBOSE: print(f'Frame {rec_prompt} rec_prompt: {rec_prompt}')

                if VERBOSE:
                    print(neg_prompt, 'neg_prompt')
                    print('init_scale pre sd run', init_scale)
                # init_latent_scale = args.init_latent_scale
                # if frame_num>0:
                #   init_latent_scale = args.frames_latent_scale
                steps = int(get_scheduled_arg(frame_num, main_config.steps_schedule))
                init_scale = get_scheduled_arg(frame_num, main_config.init_scale_schedule)
                init_latent_scale = get_scheduled_arg(frame_num, main_config.latent_scale_schedule)
                style_strength = get_scheduled_arg(frame_num, main_config.style_strength_schedule)
                skip_steps = int(steps - steps * style_strength)
                cfg_scale = get_scheduled_arg(frame_num, main_config.cfg_scale_schedule)
                image_scale = get_scheduled_arg(frame_num, main_config.image_scale_schedule)
                if VERBOSE: printf('skip_steps b4 run_sd: ', skip_steps)

                deflicker_src = {
                    'processed1': f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png',
                    'raw1': f'{video_config.videoFramesFolder}/{frame_num:06}.jpg',
                    'raw2': f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg',
                }

                init_grad_img = None
                if main_config.init_grad: init_grad_img = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'
                # setup depth source
                if main_config.cond_image_src == 'init':
                    cond_image = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'
                if main_config.cond_image_src == 'stylized':
                    cond_image = init_image
                if main_config.cond_image_src == 'cond_video':
                    cond_image = f'{video_config.condVideoFramesFolder}/{frame_num + 1:06}.jpg'

                ref_image = None
                if ref_config.reference_source == 'init':
                    ref_image = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'
                if ref_config.reference_source == 'stylized':
                    ref_image = init_image
                if ref_config.reference_source == 'prev_frame':
                    ref_image = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                if ref_config.reference_source == 'color_video':
                    if os.path.exists(f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                        ref_image = f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                    elif os.path.exists(f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'):
                        ref_image = f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'
                    else:
                        raise Exception("Reference mode specified with no color video or image. Please specify color video or disable the shuffle model")

                # setup shuffle
                shuffle_source = None
                if 'control_sd15_shuffle' in main_config.controlnet_multimodel.keys():
                    if main_config.control_sd15_shuffle_source == 'color_video':
                        if os.path.exists(f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                            shuffle_source = f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                        elif os.path.exists(f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'):
                            shuffle_source = f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'
                        else:
                            raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
                    elif main_config.control_sd15_shuffle_source == 'init':
                        shuffle_source = init_image
                    elif main_config.control_sd15_shuffle_source == 'first_frame':
                        shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{0:06}.png'
                    elif main_config.control_sd15_shuffle_source == 'prev_frame':
                        shuffle_source = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                    if not os.path.exists(shuffle_source):
                        if main_config.control_sd15_shuffle_1st_source == 'init':
                            shuffle_source = init_image
                        elif main_config.control_sd15_shuffle_1st_source == None:
                            shuffle_source = None
                        elif main_config.control_sd15_shuffle_1st_source == 'color_video':
                            if os.path.exists(f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'):
                                shuffle_source = f'{video_config.colorVideoFramesFolder}/{frame_num + 1:06}.jpg'
                            elif os.path.exists(f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'):
                                shuffle_source = f'{video_config.colorVideoFramesFolder}/{1:06}.jpg'
                            else:
                                raise Exception("Shuffle controlnet specified with no color video or image. Please specify color video or disable the shuffle model")
                    print('Shuffle source ', shuffle_source)

                prev_frame = ''
                # setup temporal source
                if main_config.temporalnet_source == 'init':
                    prev_frame = f'{video_config.videoFramesFolder}/{frame_num:06}.jpg'
                if main_config.temporalnet_source == 'stylized':
                    prev_frame = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - 1:06}.png'
                if main_config.temporalnet_source == 'cond_video':
                    prev_frame = f'{video_config.condVideoFramesFolder}/{frame_num:06}.jpg'
                if not os.path.exists(prev_frame):
                    if main_config.temporalnet_skip_1st_frame:
                        print('prev_frame not found, replacing 1st videoframe init')
                        prev_frame = None
                    else:
                        prev_frame = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'

                # setup rec noise source
                if main_config.rec_source == 'stylized':
                    rec_frame = init_image
                elif main_config.rec_source == 'init':
                    rec_frame = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'

                # setup masks for inpainting model
                from scripts.model_process.model_env import model_version
                if model_version == 'v1_inpainting':
                    if main_config.inpainting_mask_source == 'consistency_mask':
                        cond_image = consistency_mask
                    if main_config.inpainting_mask_source in ['none', None, '', 'None', 'off']:
                        cond_image = None
                    if main_config.inpainting_mask_source == 'cond_video': cond_image = f'{video_config.condVideoFramesFolder}/{frame_num + 1:06}.jpg'
                    # print('cond_image0',cond_image)

                # setup masks for controlnet inpainting model
                control_inpainting_mask = None
                if 'control_sd15_inpaint' in main_config.controlnet_multimodel.keys():
                    if main_config.control_sd15_inpaint_mask_source == 'consistency_mask':
                        control_inpainting_mask = consistency_mask
                    if main_config.control_sd15_inpaint_mask_source in ['none', None, '', 'None', 'off']:
                        # control_inpainting_mask = None
                        control_inpainting_mask = np.ones((args.side_y, args.side_x, 3))
                    if main_config.control_sd15_inpaint_mask_source == 'cond_video':
                        control_inpainting_mask = f'{video_config.condVideoFramesFolder}/{frame_num + 1:06}.jpg'
                        control_inpainting_mask = np.array(PIL.Image.open(control_inpainting_mask))
                        # print('cond_image0',cond_image)

                np_alpha = None
                if main_config.alpha_masked_diffusion and frame_num > args.start_frame:
                    if VERBOSE: print('Using alpha masked diffusion')
                    print(f'{video_config.videoFramesAlpha}/{frame_num + 1:06}.jpg')
                    if video_config.videoFramesAlpha == video_config.videoFramesFolder or not os.path.exists(f'{video_config.videoFramesAlpha}/{frame_num + 1:06}.jpg'):
                        raise Exception(
                            'You have enabled alpha_masked_diffusion without providing an alpha mask source. Please go to mask cell and specify a masked video init or extract a mask from init video.')

                    init_image_alpha = Image.open(f'{video_config.videoFramesAlpha}/{frame_num + 1:06}.jpg').resize((args.side_x, args.side_y)).convert('L')
                    np_alpha = np.array(init_image_alpha) / 255.

                sample, latent, depth_img = run_sd(args,
                                                   init_image=init_image,
                                                   skip_timesteps=skip_steps,
                                                   H=args.side_y,
                                                   W=args.side_x,
                                                   text_prompt=text_prompt,
                                                   neg_prompt=neg_prompt,
                                                   steps=steps,
                                                   seed=seed,
                                                   init_scale=init_scale,
                                                   init_latent_scale=init_latent_scale,
                                                   cond_image=cond_image,
                                                   cfg_scale=cfg_scale,
                                                   image_scale=image_scale,
                                                   config=main_config,
                                                   video_config=video_config,
                                                   ref_config=ref_config,
                                                   model_config=model_config,
                                                   content_config=content_config,
                                                   clip_config=clip_config,
                                                   sd_model=sd_model,
                                                   cond_fn=None,
                                                   init_grad_img=init_grad_img,
                                                   consistency_mask=consistency_mask,
                                                   frame_num=frame_num,
                                                   deflicker_src=deflicker_src,
                                                   prev_frame=prev_frame,
                                                   rec_prompt=rec_prompt,
                                                   rec_frame=rec_frame,
                                                   control_inpainting_mask=control_inpainting_mask,
                                                   shuffle_source=shuffle_source,
                                                   ref_image=ref_image,
                                                   alpha_mask=np_alpha)

                # settings_json = save_settings(main_config,skip_save=True)
                # settings_exif = json2exif(settings_json)
                settings_exif = json2exif('{"auther":"toto"}')

            # depth_img.save(f'{root_dir}/depth_{frame_num}.png')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            # if warp_mode == 'use_raw':torch.save(sample,f'{batchFolder}/{filename[:-4]}_raw.pt')
            if main_config.warp_mode == 'use_latent':
                torch.save(latent, f'{batchFolder}/{filename[:-4]}_lat.pt')
            samples = sample * (steps - skip_steps)
            samples = [{"pred_xstart": sample} for sample in samples]
            # for j, sample in enumerate(samples):
            # print(j, sample["pred_xstart"].size)
            # raise Exception
            if VERBOSE: print(sample[0][0].shape)
            image = sample[0][0]
            if main_config.do_softcap:
                image = softcap(image, thresh=main_config.softcap_thresh, q=main_config.softcap_q)
            image = image.add(1).div(2).clamp(0, 1)
            image = TF.to_pil_image(image)
            if main_config.warp_towards_init != 'off' and frame_num != 0:
                if main_config.warp_towards_init == 'init':
                    warp_init_filename = f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg'
                else:
                    warp_init_filename = init_image
                print('warping towards init')
                init_pil = Image.open(warp_init_filename)
                image = warp_towards_init_fn(image, init_pil, main_config, video_config.flow_lq)

            display.clear_output(wait=True)
            fit(image, main_config.display_size).save('progress.png', exif=settings_exif)
            display.display(display.Image('progress.png'))

            if main_config.mask_result and main_config.check_consistency and frame_num > 0:

                if VERBOSE: print('imitating inpaint')
                frame1_path = f'{video_config.videoFramesFolder}/{frame_num:06}.jpg'
                weights_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
                consistency_mask = load_cc(main_config, weights_path, blur=main_config.consistency_blur, dilate=main_config.consistency_dilate)

                consistency_mask = cv2.GaussianBlur(consistency_mask, (main_config.diffuse_inpaint_mask_blur, main_config.diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
                if main_config.diffuse_inpaint_mask_thresh < 1:
                    consistency_mask = np.where(consistency_mask < main_config.diffuse_inpaint_mask_thresh, 0, 1.)
                # if dither:
                #   consistency_mask = Dither.dither(consistency_mask, 'simple2D', resize=True)

                # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
                if main_config.warp_mode == 'use_image':
                    consistency_mask = cv2.GaussianBlur(consistency_mask, (3, 3), cv2.BORDER_DEFAULT)
                    init_img_prev = Image.open(init_image)
                    if VERBOSE: print(init_img_prev.size, consistency_mask.shape, image.size)
                    cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                    image_masked = np.array(image) * (1 - consistency_mask) + np.array(init_img_prev) * (consistency_mask)

                    # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                    image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                    # image = image_masked.resize(image.size, warp_interp)
                    image = image_masked
                if main_config.warp_mode == 'use_latent':
                    if main_config.invert_mask: consistency_mask = 1 - consistency_mask
                    init_lat_prev = torch.load('prevFrameScaled_lat.pt')
                    sample_masked = sd_model.decode_first_stage(latent.cuda())[0]
                    image_prev = TF.to_pil_image(sample_masked.add(1).div(2).clamp(0, 1))

                    cc_small = consistency_mask[::8, ::8, 0]
                    latent = latent.cpu() * (1 - cc_small) + init_lat_prev * cc_small
                    torch.save(latent, 'prevFrameScaled_lat.pt')

                    # image_prev = Image.open(f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num-1:06}.png')
                    torch.save(latent, 'prevFrame_lat.pt')
                    # cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
                    # image_prev = Image.open('prevFrameScaled.png')
                    image_masked = np.array(image) * (1 - consistency_mask) + np.array(image_prev) * (consistency_mask)

                    # # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
                    image_masked = Image.fromarray(image_masked.round().astype('uint8'))
                    # image = image_masked.resize(image.size, warp_interp)
                    image = image_masked

            if (frame_num > args.start_frame) or ('color_video' in main_config.normalize_latent):
                if 'frame' in main_config.normalize_latent:
                    def img2latent(img_path):
                        frame2 = Image.open(img_path)
                        frame2pil = frame2.convert('RGB').resize(image.size, main_config.warp_interp)
                        frame2pil = np.array(frame2pil)
                        frame2pil = (frame2pil / 255.)[None, ...].transpose(0, 3, 1, 2)
                        frame2pil = 2 * torch.from_numpy(frame2pil).float().cuda() - 1.
                        frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
                        return frame2pil

                    try:
                        if VERBOSE: print('Matching latent to:')
                        filename = get_frame_from_color_mode(main_config.normalize_latent, main_config.normalize_latent_offset, frame_num, video_config, args)
                        match_latent = img2latent(filename)
                        first_latent = match_latent
                        first_latent_source = filename
                        # print(first_latent_source, first_latent)
                    except:
                        if VERBOSE: print(traceback.format_exc())
                        print(f'Frame with offset/position {main_config.normalize_latent_offset} not found')
                        if 'init' in main_config.normalize_latent:
                            try:
                                filename = f'{video_config.videoFramesFolder}/{0:06}.jpg'
                                match_latent = img2latent(filename)
                                first_latent = match_latent
                                first_latent_source = filename
                            except:
                                pass
                        print(f'Color matching the 1st frame.')

                if main_config.colormatch_frame != 'off' and main_config.colormatch_after:
                    if not main_config.turbo_mode & (frame_num % int(main_config.turbo_steps) != 0) or main_config.colormatch_turbo:
                        try:
                            print('Matching color to:')
                            filename = get_frame_from_color_mode(main_config.colormatch_frame, main_config.colormatch_offset, frame_num, video_config, args)
                            match_frame = Image.open(filename)
                            first_frame = match_frame
                            first_frame_source = filename

                        except:
                            print(f'Frame with offset/position {main_config.colormatch_offset} not found')
                            if 'init' in main_config.colormatch_frame:
                                try:
                                    filename = f'{video_config.videoFramesFolder}/{1:06}.jpg'
                                    match_frame = Image.open(filename)
                                    first_frame = match_frame
                                    first_frame_source = filename
                                except:
                                    pass
                            print(f'Color matching the 1st frame.')
                        print('Colormatch source - ', first_frame_source)
                        image = Image.fromarray(
                            match_color_var(first_frame, image, opacity=main_config.color_match_frame_str, f=main_config.colormatch_method_fn, regrain=main_config.colormatch_regrain))

            if frame_num == args.start_frame:
                # settings_json = save_setings()
                pass
            if args.animation_mode != "None":
                # sys.exit(os.getcwd(), 'cwd')
                if main_config.warp_mode == 'use_image':
                    image.save('prevFrame.png', exif=settings_exif)
                else:
                    torch.save(latent, 'prevFrame_lat.pt')
            filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
            image.save(f'{batchFolder}/{filename}', exif=settings_exif)
            # np.save(latent, f'{batchFolder}/{filename[:-4]}.npy')
            if args.animation_mode == 'Video Input':
                # If turbo, save a blended image
                if main_config.turbo_mode and frame_num > args.start_frame:
                    # Mix new image with prevFrameScaled
                    blend_factor = (1) / int(main_config.turbo_steps)
                    if main_config.warp_mode == 'use_image':
                        newFrame = cv2.imread('prevFrame.png')  # This is already updated..
                        prev_frame_warped = cv2.imread('prevFrameScaled.png')
                        blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1 - blend_factor), 0.0)
                        cv2.imwrite(f'{batchFolder}/{filename}', blendedImage)
                    if main_config.warp_mode == 'use_latent':
                        newFrame = torch.load('prevFrame_lat.pt').cuda()
                        prev_frame_warped = torch.load('prevFrameScaled_lat.pt').cuda()
                        blendedImage = newFrame * (blend_factor) + prev_frame_warped * (1 - blend_factor)
                        blendedImage = get_image_from_lat(blendedImage, sd_model)
                        blendedImage.save(f'{batchFolder}/{filename}', exif=settings_exif)

            else:
                image.save(f'{batchFolder}/{filename}', exif=settings_exif)
                image.save('prevFrameScaled.png', exif=settings_exif)

        plt.plot(np.array(loss_values), 'r')
    batchBar.close()

def get_scheduled_arg(frame_num, schedule):
    global blend_json_schedules
    if isinstance(schedule, list):
        return schedule[frame_num] if frame_num < len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
        return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)

def get_sched_from_json(frame_num, sched_json, blend=False):
    frame_num = max(frame_num, 0)
    sched_int = {}
    for key in sched_json.keys():
        sched_int[int(key)] = sched_json[key]
    sched_json = sched_int
    keys = sorted(list(sched_json.keys()))
    # print(keys)
    if frame_num < 0:
        frame_num = max(keys)
    try:
        frame_num = min(frame_num, max(keys))  # clamp frame num to 0:max(keys) range
    except:
        pass

    # print('clamped frame num ', frame_num)
    if frame_num in keys:
        return sched_json[frame_num]
        # print('frame in keys')
    if frame_num not in keys:
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i + 1]
            if frame_num > k1 and frame_num < k2:
                if not blend:
                    print('frame between keys, no blend')
                    return sched_json[k1]
                if blend:
                    total_dist = k2 - k1
                    dist_from_k1 = frame_num - k1
                    return sched_json[k1] * (1 - dist_from_k1 / total_dist) + sched_json[k2] * (dist_from_k1 / total_dist)
            # else: print(f'frame {frame_num} not in {k1} {k2}')
    return 0

def do_3d_step(img_filepath, frame_num, main_config: MainConfig, video_config: VideoConfig, sd_model, forward_clip):
    if main_config.warp_mode == 'use_image':
        prev = Image.open(img_filepath)
    # if main_config.warp_mode == 'use_latent':
    #   prev = torch.load(img_filepath[:-4]+'_lat.pt')

    frame1_path = f'{video_config.videoFramesFolder}/{frame_num:06}.jpg'
    frame2 = Image.open(f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg')

    flo_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}.npy"

    if main_config.flow_override_map not in [[], '', None]:
        mapped_frame_num = int(get_scheduled_arg(frame_num, main_config.flow_override_map))
        frame_override_path = f'{video_config.videoFramesFolder}/{mapped_frame_num:06}.jpg'
        flo_path = f"{video_config.flo_folder}/{frame_override_path.split('/')[-1]}.npy"

    if main_config.use_background_mask and not main_config.apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        # else:
        if VERBOSE: print('creating bg mask for frame ', frame_num)
        frame2 = apply_mask(frame2, frame_num, main_config.background, main_config.background_source, main_config, video_config, sd_model)
        # frame2.save(f'frame2_{frame_num}.jpg')
    # init_image = 'warped.png'
    flow_blend = get_scheduled_arg(frame_num, main_config.flow_blend_schedule)
    printf('flow_blend: ', flow_blend, 'frame_num:', frame_num, 'len(flow_blend_schedule):', len(main_config.flow_blend_schedule))
    weights_path = None
    forward_clip = main_config.forward_weights_clip
    if main_config.check_consistency:
        if main_config.args.reverse_cc_order:
            weights_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
        else:
            weights_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

    if main_config.turbo_mode & (frame_num % int(main_config.turbo_steps) != 0):
        if main_config.forward_weights_clip_turbo_step:
            forward_clip = main_config.forward_weights_clip_turbo_step
        if main_config.disable_cc_for_turbo_frames:
            if VERBOSE: print('disabling cc for turbo frames')
            weights_path = None
    if main_config.warp_mode == 'use_image':
        prev = Image.open(img_filepath)

        if not main_config.warp_forward:
            printf('warping')
            warped = warp(main_config, prev,
                          frame2,
                          flo_path,
                          blend=flow_blend,
                          weights_path=weights_path,
                          forward_clip=forward_clip,
                          pad_pct=main_config.padding_ratio,
                          padding_mode=main_config.padding_mode,
                          inpaint_blend=main_config.inpaint_blend,
                          warp_mul=main_config.warp_strength)
        else:
            flo_path = f"{video_config.flo_folder}/{frame1_path.split('/')[-1]}_12.npy"
            flo = np.load(flo_path)
            warped = k_means_warp(flo, prev, main_config.warp_num_k)
        if main_config.colormatch_frame != 'off' and not main_config.colormatch_after:
            if not main_config.turbo_mode & (frame_num % int(main_config.turbo_steps) != 0) or main_config.colormatch_turbo:
                try:
                    print('Matching color before warp to:')
                    filename = get_frame_from_color_mode(main_config.colormatch_frame, main_config.colormatch_offset, frame_num, video_config, main_config.args)
                    match_frame = Image.open(filename)
                    first_frame = match_frame
                    first_frame_source = filename

                except:
                    print(traceback.format_exc())
                    print(f'Frame with offset/position {main_config.colormatch_offset} not found')
                    if 'init' in main_config.colormatch_frame:
                        try:
                            filename = f'{video_config.videoFramesFolder}/{1:06}.jpg'
                            match_frame = Image.open(filename)
                            first_frame = match_frame
                            first_frame_source = filename
                        except:
                            pass
                print(f'Color matching the 1st frame before warp.')
                print('Colormatch source - ', first_frame_source)
                warped = Image.fromarray(match_color_var(first_frame, warped, opacity=main_config.color_match_frame_str, f=main_config.colormatch_method_fn, regrain=main_config.colormatch_regrain))
    if main_config.warp_mode == 'use_latent':
        prev = torch.load(img_filepath[:-4] + '_lat.pt')
        warped = warp_lat(main_config, sd_model, prev,
                          frame2,
                          flo_path,
                          blend=flow_blend,
                          weights_path=weights_path,
                          forward_clip=forward_clip,
                          pad_pct=main_config.padding_ratio,
                          padding_mode=main_config.padding_mode,
                          inpaint_blend=main_config.inpaint_blend,
                          warp_mul=main_config.warp_strength)
    # warped = warped.resize((side_x,side_y), warp_interp)

    if main_config.use_background_mask and main_config.apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        #   return warped
        if VERBOSE: print('creating bg mask for frame ', frame_num)
        if main_config.warp_mode == 'use_latent':
            warped = apply_mask(warped, frame_num, main_config.background, main_config.background_source, main_config.invert_mask, main_config.warp_mode)
        else:
            warped = apply_mask(warped, frame_num, main_config.background, main_config.background_source, main_config.invert_mask, main_config.warp_mode)
        # warped.save(f'warped_{frame_num}.jpg')

    return warped

def printf(*msg, file=f'{root_dir}/log.txt'):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(file, 'a') as f:
        msg = f'{dt_string}> {" ".join([str(o) for o in (msg)])}'
        print(msg, file=f)

def get_frame_from_color_mode(mode, offset, frame_num, video_config: VideoConfig, args):
    if mode == 'color_video':
        if VERBOSE: print(f'the color video frame number {offset}.')
        filename = f'{video_config.colorVideoFramesFolder}/{offset + 1:06}.jpg'
    if mode == 'color_video_offset':
        if VERBOSE: print(f'the color video frame with offset {offset}.')
        filename = f'{video_config.colorVideoFramesFolder}/{frame_num - offset + 1:06}.jpg'
    if mode == 'stylized_frame_offset':
        if VERBOSE: print(f'the stylized frame with offset {offset}.')
        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - offset:06}.png'
    if mode == 'stylized_frame':
        if VERBOSE: print(f'the stylized frame number {offset}.')
        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png'
        if not os.path.exists(filename):
            filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{args.start_frame + offset:06}.png'
    if mode == 'init_frame_offset':
        if VERBOSE: print(f'the raw init frame with offset {offset}.')
        filename = f'{video_config.videoFramesFolder}/{frame_num - offset + 1:06}.jpg'
    if mode == 'init_frame':
        if VERBOSE: print(f'the raw init frame number {offset}.')
        filename = f'{video_config.videoFramesFolder}/{offset + 1:06}.jpg'
    return filename

def get_image_from_lat(lat, sd_model):
    img = sd_model.decode_first_stage(lat.cuda())[0]
    return TF.to_pil_image(img.add(1).div(2).clamp(0, 1))

def apply_mask(
        init_image,
        frame_num,
        background,
        background_source,
        main_config: MainConfig,
        video_config: VideoConfig,
        sd_model,
        invert_mask=False,
        warp_mode='use_image',
):
    if warp_mode == 'use_image':
        size = init_image.size
    if warp_mode == 'use_latent':
        print(init_image.shape)
        size = init_image.shape[-1], init_image.shape[-2]
        size = [o * 8 for o in size]
        print('size', size)
    init_image_alpha = Image.open(f'{video_config.videoFramesAlpha}/{frame_num + 1:06}.jpg').resize(size).convert('L')
    if invert_mask:
        init_image_alpha = ImageOps.invert(init_image_alpha)
    if main_config.mask_clip[1] < 255 or main_config.mask_clip[0] > 0:
        arr = np.array(init_image_alpha)
        if main_config.mask_clip[1] < 255:
            arr = np.where(arr < main_config.mask_clip[1], arr, 255)
        if main_config.mask_clip[0] > 0:
            arr = np.where(arr > main_config.mask_clip[0], arr, 0)
        init_image_alpha = Image.fromarray(arr)

    if background == 'color':
        bg = Image.new('RGB', size, background_source)
    if background == 'image':
        bg = Image.open(background_source).convert('RGB').resize(size)
    if background == 'init_video':
        bg = Image.open(f'{video_config.videoFramesFolder}/{frame_num + 1:06}.jpg').resize(size)
    # init_image.putalpha(init_image_alpha)
    if warp_mode == 'use_image':
        bg.paste(init_image, (0, 0), init_image_alpha)
    if warp_mode == 'use_latent':
        # convert bg to latent

        bg = np.array(bg)
        bg = (bg / 255.)[None, ...].transpose(0, 3, 1, 2)
        bg = 2 * torch.from_numpy(bg).float().cuda() - 1.
        bg = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(bg))
        bg = bg.cpu().numpy()  # [0].transpose(1,2,0)
        init_image_alpha = np.array(init_image_alpha)[::8, ::8][None, None, ...]
        init_image_alpha = np.repeat(init_image_alpha, 4, axis=1) / 255
        print(bg.shape, init_image.shape, init_image_alpha.shape, init_image_alpha.max(), init_image_alpha.min())
        bg = init_image * init_image_alpha + bg * (1 - init_image_alpha)
    return bg

def regen_perlin(prelin_mode, batch_size, side_x, side_y):
    if prelin_mode == 'color':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], side_x, side_y, 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], side_x, side_y, 4, 4, False)
    elif prelin_mode == 'gray':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], side_x, side_y, 1, 1, True)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], side_x, side_y, 4, 4, True)
    else:
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], side_x, side_y, 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], side_x, side_y, 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)

def create_perlin_noise(octaves=[1, 1, 1, 1], side_x=0, side_y=0, width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def interp(t):
    return 3 * t ** 2 - 2 * t ** 3

def save_settings(main_config: MainConfig, skip_save=False):
    settings_out = batchFolder + f"/settings"
    os.makedirs(settings_out, exist_ok=True)
    setting_list = {
        'text_prompts': main_config.text_prompts,
        'user_comment': main_config.user_comment,
        'image_prompts': main_config.image_prompts,
        'range_scale': main_config.range_scale,
        'sat_scale': main_config.sat_scale,
        'max_frames': main_config.max_frames,
        'interp_spline': main_config.interp_spline,
        'init_image': main_config.init_image,
        'clamp_grad': main_config.clamp_grad,
        'clamp_max': main_config.clamp_max,
        'seed': main_config.seed,
        'width': main_config.width_height[0],
        'height': main_config.width_height[1],
        'diffusion_model': diffusion_model,
        'diffusion_steps': main_config.diffusion_steps,
        'max_frames': main_config.max_frames,
        'video_init_path': main_config.video_init_path,
        'extract_nth_frame': main_config.extract_nth_frame,
        'flow_video_init_path': main_config.flow_video_init_path,
        'flow_extract_nth_frame': main_config.flow_extract_nth_frame,
        'video_init_seed_continuity': main_config.video_init_seed_continuity,
        'turbo_mode': main_config.turbo_mode,
        'turbo_steps': main_config.turbo_steps,
        'turbo_preroll': main_config.turbo_preroll,
        'flow_warp': main_config.flow_warp,
        'check_consistency': main_config.check_consistency,
        'turbo_frame_skips_steps': main_config.turbo_frame_skips_steps,
        'forward_weights_clip': main_config.forward_weights_clip,
        'forward_weights_clip_turbo_step': main_config.forward_weights_clip_turbo_step,
        'padding_ratio': main_config.padding_ratio,
        'padding_mode': main_config.padding_mode,
        'consistency_blur': main_config.consistency_blur,
        'inpaint_blend': main_config.inpaint_blend,
        'match_color_strength': main_config.match_color_strength,
        'high_brightness_threshold': main_config.high_brightness_threshold,
        'high_brightness_adjust_ratio': main_config.high_brightness_adjust_ratio,
        'low_brightness_threshold': main_config.low_brightness_threshold,
        'low_brightness_adjust_ratio': main_config.low_brightness_adjust_ratio,
        'stop_early': main_config.stop_early,
        'high_brightness_adjust_fix_amount': main_config.high_brightness_adjust_fix_amount,
        'low_brightness_adjust_fix_amount': main_config.low_brightness_adjust_fix_amount,
        'max_brightness_threshold': main_config.max_brightness_threshold,
        'min_brightness_threshold': main_config.min_brightness_threshold,
        'enable_adjust_brightness': main_config.enable_adjust_brightness,
        'dynamic_thresh': main_config.dynamic_thresh,
        'warp_interp': main_config.warp_interp,
        'fixed_code': main_config.fixed_code,
        'code_randomness': main_config.code_randomness,
        # 'normalize_code': normalize_code,
        'mask_result': main_config.mask_result,
        'reverse_cc_order': main_config.reverse_cc_order,
        'flow_lq': main_config.flow_lq,
        'use_predicted_noise': main_config.use_predicted_noise,
        'clip_guidance_scale': main_config.clip_guidance_scale,
        'clip_type': main_config.clip_type,
        'clip_pretrain': main_config.clip_pretrain,
        'missed_consistency_weight': main_config.missed_consistency_weight,
        'overshoot_consistency_weight': main_config.overshoot_consistency_weight,
        'edges_consistency_weight': main_config.edges_consistency_weight,
        'style_strength_schedule': main_config.style_strength_schedule_bkup,
        'flow_blend_schedule': main_config.flow_blend_schedule_bkup,
        'steps_schedule': main_config.steps_schedule_bkup,
        'init_scale_schedule': main_config.init_scale_schedule_bkup,
        'latent_scale_schedule': main_config.latent_scale_schedule_bkup,
        'latent_scale_template': main_config.latent_scale_template,
        'init_scale_template': main_config.init_scale_template,
        'steps_template': main_config.steps_template,
        'style_strength_template': main_config.style_strength_template,
        'flow_blend_template': main_config.flow_blend_template,
        'make_schedules': main_config.make_schedules,
        'normalize_latent': main_config.normalize_latent,
        'normalize_latent_offset': main_config.normalize_latent_offset,
        'colormatch_frame': main_config.colormatch_frame,
        'use_karras_noise': main_config.use_karras_noise,
        'end_karras_ramp_early': main_config.end_karras_ramp_early,
        'use_background_mask': main_config.use_background_mask,
        'apply_mask_after_warp': main_config.apply_mask_after_warp,
        'background': main_config.background,
        'background_source': main_config.background_source,
        'mask_source': main_config.mask_source,
        'extract_background_mask': main_config.extract_background_mask,
        'mask_video_path': main_config.mask_video_path,
        'negative_prompts': main_config.negative_prompts,
        'invert_mask': main_config.invert_mask,
        'warp_strength': main_config.warp_strength,
        'flow_override_map': main_config.flow_override_map,
        'cfg_scale_schedule': main_config.cfg_scale_schedule_bkup,
        'respect_sched': main_config.respect_sched,
        'color_match_frame_str': main_config.color_match_frame_str,
        'colormatch_offset': main_config.colormatch_offset,
        'latent_fixed_mean': main_config.latent_fixed_mean,
        'latent_fixed_std': main_config.latent_fixed_std,
        'colormatch_method': main_config.colormatch_method,
        'colormatch_regrain': main_config.colormatch_regrain,
        'warp_mode': main_config.warp_mode,
        'use_patchmatch_inpaiting': main_config.use_patchmatch_inpaiting,
        'blend_latent_to_init': main_config.blend_latent_to_init,
        'warp_towards_init': main_config.warp_towards_init,
        'init_grad': main_config.init_grad,
        'grad_denoised': main_config.grad_denoised,
        'colormatch_after': main_config.colormatch_after,
        'colormatch_turbo': main_config.colormatch_turbo,
        'model_version': main_config.model_version,
        'cond_image_src': main_config.cond_image_src,
        'warp_num_k': main_config.warp_num_k,
        'warp_forward': main_config.warp_forward,
        'sampler': main_config.sampler.__name__,
        'mask_clip': (main_config.mask_clip_low, main_config.mask_clip_high),
        'inpainting_mask_weight': main_config.inpainting_mask_weight,
        'inverse_inpainting_mask': main_config.inverse_inpainting_mask,
        'mask_source': main_config.mask_source,
        'model_path': main_config.model_path,
        'diff_override': main_config.diff_override,
        'image_scale_schedule': main_config.image_scale_schedule_bkup,
        'image_scale_template': main_config.image_scale_template,
        'frame_range': main_config.frame_range,
        'detect_resolution': main_config.detect_resolution,
        'bg_threshold': main_config.bg_threshold,
        'diffuse_inpaint_mask_blur': main_config.diffuse_inpaint_mask_blur,
        'diffuse_inpaint_mask_thresh': main_config.diffuse_inpaint_mask_thresh,
        'add_noise_to_latent': main_config.add_noise_to_latent,
        'noise_upscale_ratio': main_config.noise_upscale_ratio,
        'fixed_seed': main_config.fixed_seed,
        'init_latent_fn': spherical_dist_loss.__name__,
        'value_threshold': main_config.value_threshold,
        'distance_threshold': main_config.distance_threshold,
        'masked_guidance': main_config.masked_guidance,
        'cc_masked_diffusion': main_config.cc_masked_diffusion,
        'alpha_masked_diffusion': main_config.alpha_masked_diffusion,
        'inverse_mask_order': main_config.inverse_mask_order,
        'invert_alpha_masked_diffusion': main_config.invert_alpha_masked_diffusion,
        'quantize': main_config.quantize,
        'cb_noise_upscale_ratio': main_config.cb_noise_upscale_ratio,
        'cb_add_noise_to_latent': main_config.cb_add_noise_to_latent,
        'cb_use_start_code': main_config.cb_use_start_code,
        'cb_fixed_code': main_config.cb_fixed_code,
        'cb_norm_latent': main_config.cb_norm_latent,
        'guidance_use_start_code': main_config.guidance_use_start_code,
        'offload_model': main_config.offload_model,
        'controlnet_preprocess': main_config.controlnet_preprocess,
        'small_controlnet_model_path': main_config.small_controlnet_model_path,
        'use_scale': main_config.use_scale,
        'g_invert_mask': main_config.g_invert_mask,
        'controlnet_multimodel': json.dumps(main_config.controlnet_multimodel),
        'img_zero_uncond': main_config.img_zero_uncond,
        'do_softcap': main_config.do_softcap,
        'softcap_thresh': main_config.softcap_thresh,
        'softcap_q': main_config.softcap_q,
        'deflicker_latent_scale': main_config.deflicker_latent_scale,
        'deflicker_scale': main_config.deflicker_scale,
        'controlnet_multimodel_mode': main_config.controlnet_multimodel_mode,
        'no_half_vae': main_config.no_half_vae,
        'temporalnet_source': main_config.temporalnet_source,
        'temporalnet_skip_1st_frame': main_config.temporalnet_skip_1st_frame,
        'rec_randomness': main_config.rec_randomness,
        'rec_source': main_config.rec_source,
        'rec_cfg': main_config.rec_cfg,
        'rec_prompts': main_config.rec_prompts,
        'inpainting_mask_source': main_config.inpainting_mask_source,
        'rec_steps_pct': main_config.rec_steps_pct,
        'max_faces': main_config.max_faces,
        'num_flow_updates': main_config.num_flow_updates,
        'control_sd15_openpose_hands_face': main_config.control_sd15_openpose_hands_face,
        'control_sd15_depth_detector': main_config.control_sd15_depth_detector,
        'control_sd15_softedge_detector': main_config.control_sd15_softedge_detector,
        'control_sd15_seg_detector': main_config.control_sd15_seg_detector,
        'control_sd15_scribble_detector': main_config.control_sd15_scribble_detector,
        'control_sd15_lineart_coarse': main_config.control_sd15_lineart_coarse,
        'control_sd15_inpaint_mask_source': main_config.control_sd15_inpaint_mask_source,
        'control_sd15_shuffle_source': main_config.control_sd15_shuffle_source,
        'control_sd15_shuffle_1st_source': main_config.control_sd15_shuffle_1st_source,
        'overwrite_rec_noise': main_config.overwrite_rec_noise,
        'use_legacy_cc': main_config.use_legacy_cc,
        'missed_consistency_dilation': main_config.missed_consistency_dilation,
        'edge_consistency_width': main_config.edge_consistency_width,
        'use_reference': main_config.use_reference,
        'reference_weight': main_config.reference_weight,
        'reference_source': main_config.reference_source,
        'reference_mode': main_config.reference_mode,
        'use_legacy_fixed_code': main_config.use_legacy_fixed_code,
        'consistency_dilate': main_config.consistency_dilate,
        'prompt_patterns_sched': main_config.prompt_patterns_sched
    }
    if not skip_save:
        try:
            settings_fname = f"{settings_out}/{batch_name}({main_config.batchNum})_settings.txt"
            if os.path.exists(settings_fname):
                s_meta = os.path.getmtime(settings_fname)
                os.rename(settings_fname, settings_fname[:-4] + str(s_meta) + '.txt')
            with open(settings_fname, "w+") as f:  # save settings
                json.dump(setting_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(e)
            print('Settings:', setting_list)
    return setting_list

def json2exif(settings):
    settings = json.dumps(settings)
    exif_ifd = {piexif.ExifIFD.UserComment: settings.encode()}
    exif_dict = {"Exif": exif_ifd}
    exif_dat = piexif.dump(exif_dict)
    return exif_dat

def softcap(arr, thresh=0.8, q=0.95):
    cap = torch.quantile(abs(arr).float(), q)
    printf('q -----', torch.quantile(abs(arr).float(), torch.Tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).cuda()))
    cap_ratio = (1 - thresh) / (cap - thresh)
    arr = torch.where(arr > thresh, thresh + (arr - thresh) * cap_ratio, arr)
    arr = torch.where(arr < -thresh, -thresh + (arr + thresh) * cap_ratio, arr)
    return arr

def warp_towards_init_fn(sample_pil, init_image, main_config: MainConfig, flow_lq):
    print('sample, init', type(sample_pil), type(init_image))
    size = sample_pil.size
    sample = img2tensor(sample_pil, main_config.warp_interp)
    init_image = img2tensor(init_image, main_config.warp_interp, size)
    flo = get_flow(init_image, sample, main_config.raft_model, half=flow_lq)
    # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
    warped = warp(main_config, sample_pil,
                  sample_pil,
                  flo_path=flo,
                  blend=1,
                  weights_path=None,
                  forward_clip=0,
                  pad_pct=main_config.padding_ratio,
                  padding_mode=main_config.padding_mode,
                  inpaint_blend=main_config.inpaint_blend,
                  warp_mul=main_config.warp_strength)
    return warped

def img2tensor(img, warp_interp, size=None):
    img = img.convert('RGB')
    if size: img = img.resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()
