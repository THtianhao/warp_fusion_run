# @title Do the Run!
# @markdown Preview max size
import copy
import gc
import os
import pathlib
import random
import sys
from glob import glob
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wget
from safetensors import safe_open

from modules import sd_hijack
from scripts.content_ware_process.content_aware_config import ContentAwareConfig
from scripts.content_ware_process.difference_fun import check_and_adjust_sched
from scripts.lora_embedding.lora_embedding_config import LoraEmbeddingConfig
from scripts.lora_embedding.lora_embedding_fun import inject_lora, split_lora_from_prompts
from scripts.model_process.model_config import ModelConfig
from scripts.model_process.model_env import model_version, control_model_urls, load_to
from scripts.run.run_env import diffusion_sampling_mode
from scripts.settings.main_config import MainConfig
from scripts.settings.setting import batch_name, batchFolder, steps, width_height, clip_guidance_scale, tv_scale, range_scale, cutn_batches, init_image, init_scale, skip_steps, side_x, side_y, \
    skip_augs
from scripts.utils.env import root_dir
from scripts.utils.path import createPath
from scripts.video_process.color_transfor_func import force_download, PT
from scripts.video_process.video_config import VideoConfig

def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora

def run_prepare_config(main_config: MainConfig,
                       model_config: ModelConfig,
                       video_config: VideoConfig,
                       lora_embedding_config: LoraEmbeddingConfig,
                       content_aware_config: ContentAwareConfig):
    sd_model = model_config.sd_mode
    torch.cuda.empty_cache()
    gc.collect()
    sd_model.control_scales = ([1] * 13)
    if model_version == 'control_multi':
        sd_model.control_model.cpu()
        print('Checking downloaded Annotator and ControlNet Models')
        for controlnet in main_config.controlnet_multimodel.keys():
            controlnet_settings = main_config.controlnet_multimodel[controlnet]
            weight = controlnet_settings["weight"]
            if weight != 0:
                small_url = control_model_urls[controlnet]
                local_filename = small_url.split('/')[-1]
                small_controlnet_model_path = f"{model_config.controlnet_models_dir}/{local_filename}"
                if model_config.use_small_controlnet and os.path.exists(model_config.model_path) and not os.path.exists(small_controlnet_model_path):
                    print(f'Model found at {model_config.model_path}. Small model not found at {small_controlnet_model_path}.')
                    if not os.path.exists(small_controlnet_model_path) or force_download:
                        try:
                            pathlib.Path(small_controlnet_model_path).unlink()
                        except:
                            pass
                        print(f'Downloading small {controlnet} model... ')
                        wget.download(small_url, small_controlnet_model_path)
                        print(f'Downloaded small {controlnet} model.')

        print('Loading ControlNet Models')
        main_config.loaded_controlnets = {}
        for controlnet in main_config.controlnet_multimodel.keys():
            controlnet_settings = main_config.controlnet_multimodel[controlnet]
            weight = controlnet_settings["weight"]
            if weight != 0:
                main_config.loaded_controlnets[controlnet] = copy.deepcopy(sd_model.control_model)
                small_url = control_model_urls[controlnet]
                local_filename = small_url.split('/')[-1]
                small_controlnet_model_path = f"{model_config.controlnet_models_dir}/{local_filename}"
                if os.path.exists(small_controlnet_model_path):
                    ckpt = small_controlnet_model_path
                    print(f"Loading model from {ckpt}")
                    if ckpt.endswith('.safetensors'):
                        pl_sd = {}
                        with safe_open(ckpt, framework="pt", device=load_to) as f:
                            for key in f.keys():
                                pl_sd[key] = f.get_tensor(key)
                    else:
                        pl_sd = torch.load(ckpt, map_location=load_to)

                    if "global_step" in pl_sd:
                        print(f"Global Step: {pl_sd['global_step']}")
                    if "state_dict" in pl_sd:
                        sd = pl_sd["state_dict"]
                    else:
                        sd = pl_sd
                    if "control_model.input_blocks.0.0.bias" in sd:
                        sd = dict([(o.split('control_model.')[-1], sd[o]) for o in sd.keys() if o != 'difference'])

                        # print('control_model in sd')
                    del pl_sd

                    gc.collect()
                    m, u = main_config.loaded_controlnets[controlnet].load_state_dict(sd, strict=True)
                    main_config.loaded_controlnets[controlnet].half()
                else:
                    print('Small controlnet model not found in path but specified in settings. Please adjust settings or check controlnet path.')
                    sys.exit(0)

    # print('Loading annotators.')
    controlnet_keys = main_config.controlnet_multimodel.keys() if model_version == 'control_multi' else model_version
    if "control_sd15_depth" in controlnet_keys or "control_sd15_normal" in controlnet_keys:
        if main_config.control_sd15_depth_detector == 'Midas' or "control_sd15_normal" in controlnet_keys:
            from annotator.midas import MidasDetector

            main_config.apply_depth = MidasDetector()
            print('Loaded MidasDetector')
        if main_config.control_sd15_depth_detector == 'Zoe':
            from annotator.zoe import ZoeDetector

            main_config.apply_depth = ZoeDetector()
            print('Loaded ZoeDetector')

    if "control_sd15_normalbae" in controlnet_keys:
        from annotator.normalbae import NormalBaeDetector

        main_config.apply_normal = NormalBaeDetector()
        print('Loaded NormalBaeDetector')
    if 'control_sd15_canny' in controlnet_keys:
        from annotator.canny import CannyDetector

        main_config.apply_canny = CannyDetector()
        print('Loaded CannyDetector')
    if 'control_sd15_softedge' in controlnet_keys:
        if main_config.control_sd15_softedge_detector == 'HED':
            from annotator.hed import HEDdetector

            main_config.apply_softedge = HEDdetector()
            print('Loaded HEDdetector')
        if main_config.control_sd15_softedge_detector == 'PIDI':
            from annotator.pidinet import PidiNetDetector

            main_config.apply_softedge = PidiNetDetector()
            print('Loaded PidiNetDetector')
    if 'control_sd15_scribble' in controlnet_keys:

        if main_config.control_sd15_scribble_detector == 'HED':
            from annotator.hed import HEDdetector

            main_config.apply_scribble = HEDdetector()
            print('Loaded HEDdetector')
        if main_config.control_sd15_scribble_detector == 'PIDI':
            from annotator.pidinet import PidiNetDetector

            main_config.apply_scribble = PidiNetDetector()
            print('Loaded PidiNetDetector')

    if "control_sd15_mlsd" in controlnet_keys:
        from annotator.mlsd import MLSDdetector

        main_config.apply_mlsd = MLSDdetector()
        print('Loaded MLSDdetector')
    if "control_sd15_openpose" in controlnet_keys:
        from annotator.openpose import OpenposeDetector

        main_config.apply_openpose = OpenposeDetector()
        print('Loaded OpenposeDetector')
    if "control_sd15_seg" in controlnet_keys:
        if main_config.control_sd15_seg_detector == 'Seg_OFCOCO':
            from annotator.oneformer import OneformerCOCODetector

            main_config.apply_seg = OneformerCOCODetector()
            print('Loaded OneformerCOCODetector')
        elif main_config.control_sd15_seg_detector == 'Seg_OFADE20K':
            from annotator.oneformer import OneformerADE20kDetector

            main_config.apply_seg = OneformerADE20kDetector()
            print('Loaded OneformerADE20kDetector')
        elif main_config.control_sd15_seg_detector == 'Seg_UFADE20K':
            from annotator.uniformer import UniformerDetector

            main_config.apply_seg = UniformerDetector()
            print('Loaded UniformerDetector')
    if "control_sd15_shuffle" in controlnet_keys:
        from annotator.shuffle import ContentShuffleDetector

        main_config.apply_shuffle = ContentShuffleDetector()
        print('Loaded ContentShuffleDetector')

    # if "control_sd15_ip2p" in controlnet_keys:
    #   #no annotator
    #   pass
    # if "control_sd15_inpaint" in controlnet_keys:
    #   #no annotator
    #   pass
    if "control_sd15_lineart" in controlnet_keys:
        from annotator.lineart import LineartDetector

        main_config.apply_lineart = LineartDetector()
        print('Loaded LineartDetector')
    if "control_sd15_lineart_anime" in controlnet_keys:
        from annotator.lineart_anime import LineartAnimeDetector

        main_config.apply_lineart_anime = LineartAnimeDetector()
        print('Loaded LineartAnimeDetector')

    unload()
    sd_model.cuda()
    sd_hijack.model_hijack.hijack(sd_model)
    sd_hijack.model_hijack.embedding_db.add_embedding_dir(lora_embedding_config.custom_embed_dir)
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(sd_model, force_reload=True)

    latent_scale_schedule_bkup = copy.copy(main_config.latent_scale_schedule)
    init_scale_schedule_bkup = copy.copy(main_config.init_scale_schedule)
    steps_schedule_bkup = copy.copy(main_config.steps_schedule)
    style_strength_schedule_bkup = copy.copy(main_config.style_strength_schedule)
    flow_blend_schedule_bkup = copy.copy(main_config.flow_blend_schedule)
    cfg_scale_schedule_bkup = copy.copy(main_config.cfg_scale_schedule)
    image_scale_schedule_bkup = copy.copy(main_config.image_scale_schedule)

    if content_aware_config.make_schedules:
        if content_aware_config.diff is None and content_aware_config.diff_override == []: sys.exit(
            f'\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous cell, run it, and then run this cell again\n')
        if content_aware_config.diff_override != []:
            diff = content_aware_config.diff_override

        print('Applied schedules:')
        main_config.latent_scale_schedule = check_and_adjust_sched(main_config.latent_scale_schedule, content_aware_config.latent_scale_template, diff, content_aware_config.respect_sched)
        main_config.init_scale_schedule = check_and_adjust_sched(main_config.init_scale_schedule, content_aware_config.init_scale_template, diff, content_aware_config.respect_sched)
        main_config.steps_schedule = check_and_adjust_sched(main_config.steps_schedule, content_aware_config.steps_template, diff, content_aware_config.respect_sched)
        main_config.style_strength_schedule = check_and_adjust_sched(main_config.style_strength_schedule, content_aware_config.style_strength_template, diff, content_aware_config.respect_sched)
        main_config.flow_blend_schedule = check_and_adjust_sched(main_config.flow_blend_schedule, content_aware_config.flow_blend_template, diff, content_aware_config.respect_sched)
        main_config.cfg_scale_schedule = check_and_adjust_sched(main_config.cfg_scale_schedule, content_aware_config.cfg_scale_template, diff, content_aware_config.respect_sched)
        main_config.image_scale_schedule = check_and_adjust_sched(main_config.image_scale_schedule, content_aware_config.cfg_scale_template, diff, content_aware_config.respect_sched)
        for sched, name in zip([main_config.latent_scale_schedule, main_config.init_scale_schedule, main_config.steps_schedule, main_config.style_strength_schedule, main_config.flow_blend_schedule,
                                main_config.cfg_scale_schedule, main_config.image_scale_schedule],
                               ['latent_scale_schedule', 'init_scale_schedule', 'steps_schedule', 'style_strength_schedule', 'flow_blend_schedule',
                                'cfg_scale_schedule', 'image_scale_schedule']):
            if type(sched) == list:
                if len(sched) > 2:
                    print(name, ': ', sched[:100])

    main_config.colormatch_method_fn = PT.lab_transfer
    if main_config.colormatch_method == 'PDF':
        main_config.colormatch_method_fn = PT.pdf_transfer
    if main_config.colormatch_method == 'mean':
        main_config.colormatch_method_fn = PT.mean_std_transfer

    if video_config.animation_mode == 'Video Input':
        main_config.max_frames = len(glob(f'{video_config.videoFramesFolder}/*.jpg'))

    def split_prompts(prompts):
        prompt_series = pd.Series([np.nan for a in range(main_config.max_frames)])
        for i, prompt in prompts.items():
            prompt_series[i] = prompt
        # prompt_series = prompt_series.astype(str)
        prompt_series = prompt_series.ffill().bfill()
        return prompt_series

    if main_config.warp_towards_init != 'off':
        if video_config.flow_lq:
            main_config.raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        else:
            main_config.raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{batch_name}({batchNum})_{i:06}.png'
            new_file = new_folder + f'/{batch_name}({batchNum})_{i:06}.png'
            os.rename(old_file, new_file)

    if main_config.retain_overwritteon_frames is True:
        retainFolder = f'{batchFolder}/retained'
        createPath(retainFolder)

    if video_config.animation_mode == 'Video Input':
        frames = sorted(glob(video_config.in_path + '/*.*'));
        if len(frames) == 0:
            sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
        flows = glob(video_config.flo_folder + '/*.*')
        if (len(flows) == 0) and video_config.flow_warp:
            sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")
    settings_out = batchFolder + f"/settings"
    if main_config.resume_run:
        if main_config.run_to_resume == 'latest':
            batchNum = len(glob(f"{settings_out}/{batch_name}(*)_settings.txt")) - 1
        else:
            batchNum = int(main_config.run_to_resume)
        if main_config.resume_from_frame == 'latest':
            start_frame = len(glob(batchFolder + f"/{batch_name}({batchNum})_*.png"))
            if video_config.animation_mode != 'Video Input' and main_config.turbo_mode == True and start_frame > main_config.turbo_preroll and start_frame % int(main_config.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(main_config.turbo_steps))
        else:
            start_frame = int(main_config.resume_from_frame) + 1
            if video_config.animation_mode != 'Video Input' and main_config.turbo_mode == True and start_frame > main_config.turbo_preroll and start_frame % int(main_config.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(main_config.turbo_steps))
            if main_config.retain_overwritten_frames is True:
                existing_frames = len(glob(batchFolder + f"/{batch_name}({batchNum})_*.png"))
                frames_to_save = existing_frames - start_frame
                print(f'Moving {frames_to_save} frames to the Retained folder')
                move_files(start_frame, existing_frames, batchFolder, retainFolder)
    else:
        start_frame = 0
        batchNum = len(glob(settings_out + "/*.txt"))
        while os.path.isfile(f"{settings_out}/{batch_name}({batchNum})_settings.txt") is True or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True:
            batchNum += 1

    print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

    if main_config.set_seed == 'random_seed' or main_config.set_seed == -1:
        random.seed()
        seed = random.randint(0, 2 ** 32)
        # print(f'Using seed: {seed}')
    else:
        seed = int(main_config.set_seed)

    main_config.new_prompt_loras = {}
    if main_config.text_prompts:
        _, main_config.new_prompt_loras = split_lora_from_prompts(main_config.text_prompts)
        print('Inferred loras schedule:\n', main_config.new_prompt_loras)

    if main_config.new_prompt_loras not in [{}, [], '', None]:
        inject_lora(sd_model)
        # load_loras(use_loras,lora_multipliers)

    main_config.args = {
        'batchNum': batchNum,
        'prompts_series': main_config.text_prompts if main_config.text_prompts else None,
        'rec_prompts_series': main_config.rec_prompts if main_config.rec_prompts else None,
        'neg_prompts_series': main_config.negative_prompts if main_config.negative_prompts else None,
        'image_prompts_series': main_config.image_prompts if main_config.image_prompts else None,

        'seed': seed,
        'display_rate': main_config.display_rate,
        'n_batches': main_config.n_batches if video_config.animation_mode == 'None' else 1,
        'batch_size': main_config.batch_size,
        'batch_name': batch_name,
        'steps': steps,
        'diffusion_sampling_mode': diffusion_sampling_mode,
        'width_height': width_height,
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'sat_scale': main_config.sat_scale,
        'cutn_batches': cutn_batches,
        'init_image': init_image,
        'init_scale': init_scale,
        'skip_steps': skip_steps,
        'side_x': side_x,
        'side_y': side_y,
        'timestep_respacing': main_config.timestep_respacing,
        'diffusion_steps': main_config.diffusion_steps,
        'animation_mode': video_config.animation_mode,
        'video_init_path': video_config.video_init_path,
        'extract_nth_frame': video_config.extract_nth_frame,
        'video_init_seed_continuity': video_config.video_init_seed_continuity,
        'key_frames': main_config.key_frames,
        'max_frames': main_config.max_frames if video_config.animation_mode != "None" else 1,
        'interp_spline': main_config.interp_spline,
        'start_frame': start_frame,
        'padding_mode': main_config.padding_mode,
        'text_prompts': main_config.text_prompts,
        'image_prompts': main_config.image_prompts,
        'intermediate_saves': main_config.intermediate_saves,
        'intermediates_in_subfolder': main_config.intermediates_in_subfolder,
        'steps_per_checkpoint': main_config.steps_per_checkpoint,
        'perlin_init': main_config.perlin_init,
        'perlin_mode': main_config.perlin_mode,
        'set_seed': main_config.set_seed,
        'clamp_grad': main_config.clamp_grad,
        'clamp_max': main_config.clamp_max,
        'skip_augs': skip_augs,
    }
    if main_config.frame_range not in [None, [0, 0], '', [0], 0]:
        main_config.args['start_frame'] = main_config.frame_range[0]
        main_config.args['max_frames'] = min(main_config.args['max_frames'], main_config.frame_range[1])
    main_config.args = SimpleNamespace(**main_config.args)
    gc.collect()
