# @title Do the Run!
# @markdown Preview max size
import gc
import os
import pathlib
import sys
from glob import glob

import torch
import wget

from modules import shared, sd_hijack
from scripts.gui.gui_env import guis
from scripts.gui.gui_func import get_value
import copy

from scripts.model_process.model_config import ModelConfig
from scripts.model_process.model_env import model_version, control_model_urls
from scripts.settings.main_settings import controlnet_multimodel
from scripts.video_process.color_transfor_func import force_download

def run_prepare_config(model_config: ModelConfig, sd_model):
    apply_depth = None
    apply_canny = None
    apply_mlsd = None
    apply_hed = None
    apply_openpose = None
    apply_seg = None
    loaded_controlnets = {}
    torch.cuda.empty_cache()
    gc.collect()
    sd_model.control_scales = ([1] * 13)
    if model_version == 'control_multi':
        sd_model.control_model.cpu()
        print('Checking downloaded Annotator and ControlNet Models')
        for controlnet in controlnet_multimodel.keys():
            controlnet_settings = controlnet_multimodel[controlnet]
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
        loaded_controlnets = {}
        for controlnet in controlnet_multimodel.keys():
            controlnet_settings = controlnet_multimodel[controlnet]
            weight = controlnet_settings["weight"]
            if weight != 0:
                loaded_controlnets[controlnet] = copy.deepcopy(sd_model.control_model)
                small_url = control_model_urls[controlnet]
                local_filename = small_url.split('/')[-1]
                small_controlnet_model_path = f"{controlnet_models_dir}/{local_filename}"
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
                    m, u = loaded_controlnets[controlnet].load_state_dict(sd, strict=True)
                    loaded_controlnets[controlnet].half()
                    if len(m) > 0 and verbose:
                        print("missing keys:")
                        print(m, len(m))
                    if len(u) > 0 and verbose:
                        print("unexpected keys:")
                        print(u, len(u))
                else:
                    print('Small controlnet model not found in path but specified in settings. Please adjust settings or check controlnet path.')
                    sys.exit(0)

    # print('Loading annotators.')
    controlnet_keys = controlnet_multimodel.keys() if model_version == 'control_multi' else model_version
    if "control_sd15_depth" in controlnet_keys or "control_sd15_normal" in controlnet_keys:
        if control_sd15_depth_detector == 'Midas' or "control_sd15_normal" in controlnet_keys:
            from annotator.midas import MidasDetector

            apply_depth = MidasDetector()
            print('Loaded MidasDetector')
        if control_sd15_depth_detector == 'Zoe':
            from annotator.zoe import ZoeDetector

            apply_depth = ZoeDetector()
            print('Loaded ZoeDetector')

    if "control_sd15_normalbae" in controlnet_keys:
        from annotator.normalbae import NormalBaeDetector

        apply_normal = NormalBaeDetector()
        print('Loaded NormalBaeDetector')
    if 'control_sd15_canny' in controlnet_keys:
        from annotator.canny import CannyDetector

        apply_canny = CannyDetector()
        print('Loaded CannyDetector')
    if 'control_sd15_softedge' in controlnet_keys:
        if control_sd15_softedge_detector == 'HED':
            from annotator.hed import HEDdetector

            apply_softedge = HEDdetector()
            print('Loaded HEDdetector')
        if control_sd15_softedge_detector == 'PIDI':
            from annotator.pidinet import PidiNetDetector

            apply_softedge = PidiNetDetector()
            print('Loaded PidiNetDetector')
    if 'control_sd15_scribble' in controlnet_keys:
        from annotator.util import nms

        if control_sd15_scribble_detector == 'HED':
            from annotator.hed import HEDdetector

            apply_scribble = HEDdetector()
            print('Loaded HEDdetector')
        if control_sd15_scribble_detector == 'PIDI':
            from annotator.pidinet import PidiNetDetector

            apply_scribble = PidiNetDetector()
            print('Loaded PidiNetDetector')

    if "control_sd15_mlsd" in controlnet_keys:
        from annotator.mlsd import MLSDdetector

        apply_mlsd = MLSDdetector()
        print('Loaded MLSDdetector')
    if "control_sd15_openpose" in controlnet_keys:
        from annotator.openpose import OpenposeDetector

        apply_openpose = OpenposeDetector()
        print('Loaded OpenposeDetector')
    if "control_sd15_seg" in controlnet_keys:
        if control_sd15_seg_detector == 'Seg_OFCOCO':
            from annotator.oneformer import OneformerCOCODetector

            apply_seg = OneformerCOCODetector()
            print('Loaded OneformerCOCODetector')
        elif control_sd15_seg_detector == 'Seg_OFADE20K':
            from annotator.oneformer import OneformerADE20kDetector

            apply_seg = OneformerADE20kDetector()
            print('Loaded OneformerADE20kDetector')
        elif control_sd15_seg_detector == 'Seg_UFADE20K':
            from annotator.uniformer import UniformerDetector

            apply_seg = UniformerDetector()
            print('Loaded UniformerDetector')
    if "control_sd15_shuffle" in controlnet_keys:
        from annotator.shuffle import ContentShuffleDetector

        apply_shuffle = ContentShuffleDetector()
        print('Loaded ContentShuffleDetector')

    # if "control_sd15_ip2p" in controlnet_keys:
    #   #no annotator
    #   pass
    # if "control_sd15_inpaint" in controlnet_keys:
    #   #no annotator
    #   pass
    if "control_sd15_lineart" in controlnet_keys:
        from annotator.lineart import LineartDetector

        apply_lineart = LineartDetector()
        print('Loaded LineartDetector')
    if "control_sd15_lineart_anime" in controlnet_keys:
        from annotator.lineart_anime import LineartAnimeDetector

        apply_lineart_anime = LineartAnimeDetector()
        print('Loaded LineartAnimeDetector')

    def deflicker_loss(processed2, processed1, raw1, raw2, criterion1, criterion2):
        raw_diff = criterion1(raw2, raw1)
        proc_diff = criterion1(processed1, processed2)
        return criterion2(raw_diff, proc_diff)

    unload()
    sd_model.cuda()
    sd_hijack.model_hijack.hijack(sd_model)
    sd_hijack.model_hijack.embedding_db.add_embedding_dir(custom_embed_dir)
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(sd_model, force_reload=True)

    latent_scale_schedule = eval(get_value('latent_scale_schedule', guis))
    init_scale_schedule = eval(get_value('init_scale_schedule', guis))
    steps_schedule = eval(get_value('steps_schedule', guis))
    style_strength_schedule = eval(get_value('style_strength_schedule', guis))
    cfg_scale_schedule = eval(get_value('cfg_scale_schedule', guis))
    flow_blend_schedule = eval(get_value('flow_blend_schedule', guis))
    image_scale_schedule = eval(get_value('image_scale_schedule', guis))

    latent_scale_schedule_bkup = copy.copy(latent_scale_schedule)
    init_scale_schedule_bkup = copy.copy(init_scale_schedule)
    steps_schedule_bkup = copy.copy(steps_schedule)
    style_strength_schedule_bkup = copy.copy(style_strength_schedule)
    flow_blend_schedule_bkup = copy.copy(flow_blend_schedule)
    cfg_scale_schedule_bkup = copy.copy(cfg_scale_schedule)
    image_scale_schedule_bkup = copy.copy(image_scale_schedule)

    if make_schedules:
        if diff is None and diff_override == []: sys.exit(f'\nERROR!\n\nframes were not anayzed. Please enable analyze_video in the previous cell, run it, and then run this cell again\n')
        if diff_override != []: diff = diff_override

        print('Applied schedules:')
        latent_scale_schedule = check_and_adjust_sched(latent_scale_schedule, latent_scale_template, diff, respect_sched)
        init_scale_schedule = check_and_adjust_sched(init_scale_schedule, init_scale_template, diff, respect_sched)
        steps_schedule = check_and_adjust_sched(steps_schedule, steps_template, diff, respect_sched)
        style_strength_schedule = check_and_adjust_sched(style_strength_schedule, style_strength_template, diff, respect_sched)
        flow_blend_schedule = check_and_adjust_sched(flow_blend_schedule, flow_blend_template, diff, respect_sched)
        cfg_scale_schedule = check_and_adjust_sched(cfg_scale_schedule, cfg_scale_template, diff, respect_sched)
        image_scale_schedule = check_and_adjust_sched(image_scale_schedule, cfg_scale_template, diff, respect_sched)
        for sched, name in zip([latent_scale_schedule, init_scale_schedule, steps_schedule, style_strength_schedule, flow_blend_schedule,
                                cfg_scale_schedule, image_scale_schedule], ['latent_scale_schedule', 'init_scale_schedule', 'steps_schedule', 'style_strength_schedule', 'flow_blend_schedule',
                                                                            'cfg_scale_schedule', 'image_scale_schedule']):
            if type(sched) == list:
                if len(sched) > 2:
                    print(name, ': ', sched[:100])

    use_karras_noise = False
    end_karras_ramp_early = False
    # use_predicted_noise = False
    warp_interp = Image.LANCZOS
    start_code_cb = None  # variable for cb_code
    guidance_start_code = None  # variable for guidance code

    display_size = 512  # @param

    user_comment = get_value('user_comment', guis)
    blend_json_schedules = get_value('blend_json_schedules', guis)
    VERBOSE = get_value('VERBOSE', guis)
    use_background_mask = get_value('use_background_mask', guis)
    invert_mask = get_value('invert_mask', guis)
    background = get_value('background', guis)
    background_source = get_value('background_source', guis)
    (mask_clip_low, mask_clip_high) = get_value('mask_clip', guis)

    # turbo
    turbo_mode = get_value('turbo_mode', guis)
    turbo_steps = get_value('turbo_steps', guis)
    colormatch_turbo = get_value('colormatch_turbo', guis)
    turbo_frame_skips_steps = get_value('turbo_frame_skips_steps', guis)
    soften_consistency_mask_for_turbo_frames = get_value('soften_consistency_mask_for_turbo_frames', guis)

    # warp
    flow_warp = get_value('flow_warp', guis)
    apply_mask_after_warp = get_value('apply_mask_after_warp', guis)
    warp_num_k = get_value('warp_num_k', guis)
    warp_forward = get_value('warp_forward', guis)
    warp_strength = get_value('warp_strength', guis)
    flow_override_map = eval(get_value('flow_override_map', guis))
    warp_mode = get_value('warp_mode', guis)
    warp_towards_init = get_value('warp_towards_init', guis)

    # cc
    check_consistency = get_value('check_consistency', guis)
    missed_consistency_weight = get_value('missed_consistency_weight', guis)
    overshoot_consistency_weight = get_value('overshoot_consistency_weight', guis)
    edges_consistency_weight = get_value('edges_consistency_weight', guis)
    consistency_blur = get_value('consistency_blur', guis)
    consistency_dilate = get_value('consistency_dilate', guis)
    padding_ratio = get_value('padding_ratio', guis)
    padding_mode = get_value('padding_mode', guis)
    match_color_strength = get_value('match_color_strength', guis)
    soften_consistency_mask = get_value('soften_consistency_mask', guis)
    mask_result = get_value('mask_result', guis)
    use_patchmatch_inpaiting = get_value('use_patchmatch_inpaiting', guis)

    # diffusion
    text_prompts = eval(get_value('text_prompts', guis))
    negative_prompts = eval(get_value('negative_prompts', guis))
    prompt_patterns_sched = eval(get_value('prompt_patterns_sched', guis))
    cond_image_src = get_value('cond_image_src', guis)
    set_seed = get_value('set_seed', guis)
    clamp_grad = get_value('clamp_grad', guis)
    clamp_max = get_value('clamp_max', guis)
    sat_scale = get_value('sat_scale', guis)
    init_grad = get_value('init_grad', guis)
    grad_denoised = get_value('grad_denoised', guis)
    blend_latent_to_init = get_value('blend_latent_to_init', guis)
    fixed_code = get_value('fixed_code', guis)
    code_randomness = get_value('code_randomness', guis)
    # normalize_code=get_value('normalize_code',guis)
    dynamic_thresh = get_value('dynamic_thresh', guis)
    sampler = get_value('sampler', guis)
    use_karras_noise = get_value('use_karras_noise', guis)
    inpainting_mask_weight = get_value('inpainting_mask_weight', guis)
    inverse_inpainting_mask = get_value('inverse_inpainting_mask', guis)
    inpainting_mask_source = get_value('mask_source', guis)

    # colormatch
    normalize_latent = get_value('normalize_latent', guis)
    normalize_latent_offset = get_value('normalize_latent_offset', guis)
    latent_fixed_mean = eval(str(get_value('latent_fixed_mean', guis)))
    latent_fixed_std = eval(str(get_value('latent_fixed_std', guis)))
    latent_norm_4d = get_value('latent_norm_4d', guis)
    colormatch_frame = get_value('colormatch_frame', guis)
    color_match_frame_str = get_value('color_match_frame_str', guis)
    colormatch_offset = get_value('colormatch_offset', guis)
    colormatch_method = get_value('colormatch_method', guis)
    colormatch_regrain = get_value('colormatch_regrain', guis)
    colormatch_after = get_value('colormatch_after', guis)
    image_prompts = {}

    fixed_seed = get_value('fixed_seed', guis)

    rec_cfg = get_value('rec_cfg', guis)
    rec_steps_pct = get_value('rec_steps_pct', guis)
    rec_prompts = eval(get_value('rec_prompts', guis))
    rec_randomness = get_value('rec_randomness', guis)
    use_predicted_noise = get_value('use_predicted_noise', guis)
    overwrite_rec_noise = get_value('overwrite_rec_noise', guis)

    # controlnet
    save_controlnet_annotations = get_value('save_controlnet_annotations', guis)
    control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face', guis)
    control_sd15_depth_detector = get_value('control_sd15_depth_detector', guis)
    control_sd15_softedge_detector = get_value('control_sd15_softedge_detector', guis)
    control_sd15_seg_detector = get_value('control_sd15_seg_detector', guis)
    control_sd15_scribble_detector = get_value('control_sd15_scribble_detector', guis)
    control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse', guis)
    control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source', guis)
    control_sd15_shuffle_source = get_value('control_sd15_shuffle_source', guis)
    control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source', guis)
    controlnet_preprocess = get_value('controlnet_preprocess', guis)

    detect_resolution = get_value('detect_resolution', guis)
    bg_threshold = get_value('bg_threshold', guis)
    low_threshold = get_value('low_threshold', guis)
    high_threshold = get_value('high_threshold', guis)
    value_threshold = get_value('value_threshold', guis)
    distance_threshold = get_value('distance_threshold', guis)
    temporalnet_source = get_value('temporalnet_source', guis)
    temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame', guis)
    controlnet_multimodel_mode = get_value('controlnet_multimodel_mode', guis)
    max_faces = get_value('max_faces', guis)

    do_softcap = get_value('do_softcap', guis)
    softcap_thresh = get_value('softcap_thresh', guis)
    softcap_q = get_value('softcap_q', guis)

    masked_guidance = get_value('masked_guidance', guis)
    cc_masked_diffusion = get_value('cc_masked_diffusion', guis)
    alpha_masked_diffusion = get_value('alpha_masked_diffusion', guis)
    invert_alpha_masked_diffusion = get_value('invert_alpha_masked_diffusion', guis)

    if turbo_frame_skips_steps == '100% (don`t diffuse turbo frames, fastest)':
        turbo_frame_skips_steps = None
    else:
        turbo_frame_skips_steps = int(turbo_frame_skips_steps.split('%')[0]) / 100

    disable_cc_for_turbo_frames = False

    colormatch_method_fn = PT.lab_transfer
    if colormatch_method == 'PDF':
        colormatch_method_fn = PT.pdf_transfer
    if colormatch_method == 'mean':
        colormatch_method_fn = PT.mean_std_transfer

    turbo_preroll = 1
    intermediate_saves = None
    intermediates_in_subfolder = True
    steps_per_checkpoint = None

    forward_weights_clip = soften_consistency_mask
    forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
    inpaint_blend = 0

    if animation_mode == 'Video Input':
        max_frames = len(glob(f'{videoFramesFolder}/*.jpg'))

    def split_prompts(prompts):
        prompt_series = pd.Series([np.nan for a in range(max_frames)])
        for i, prompt in prompts.items():
            prompt_series[i] = prompt
        # prompt_series = prompt_series.astype(str)
        prompt_series = prompt_series.ffill().bfill()
        return prompt_series

    key_frames = True
    interp_spline = 'Linear'
    perlin_init = False
    perlin_mode = 'mixed'

    if warp_towards_init != 'off':
        if flow_lq:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
        # raft_model = torch.nn.DataParallel(RAFT(args2))
        else:
            raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

    def printf(*msg, file=f'{root_dir}/log.txt'):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(file, 'a') as f:
            msg = f'{dt_string}> {" ".join([str(o) for o in (msg)])}'
            print(msg, file=f)

    printf('--------Beginning new run------')
    ##@markdown `n_batches` ignored with animation modes.
    display_rate = 9999999
    ##@param{type: 'number'}
    n_batches = 1
    ##@param{type: 'number'}
    start_code = None
    first_latent = None
    first_latent_source = 'not set'
    os.chdir(root_dir)
    n_mean_avg = None
    n_std_avg = None
    n_smooth = 0.5
    # Update Model Settings
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps

    batch_size = 1

    def move_files(start_num, end_num, old_folder, new_folder):
        for i in range(start_num, end_num):
            old_file = old_folder + f'/{batch_name}({batchNum})_{i:06}.png'
            new_file = new_folder + f'/{batch_name}({batchNum})_{i:06}.png'
            os.rename(old_file, new_file)

    noise_upscale_ratio = int(noise_upscale_ratio)
    # @markdown ---
    # @markdown Frames to run. Leave empty or [0,0] to run all frames.
    frame_range = [0, 0]  # @param
    resume_run = False  # @param{type: 'boolean'}
    run_to_resume = 'latest'  # @param{type: 'string'}
    resume_from_frame = 'latest'  # @param{type: 'string'}
    retain_overwritten_frames = False  # @param{type: 'boolean'}
    if retain_overwritten_frames is True:
        retainFolder = f'{batchFolder}/retained'
        createPath(retainFolder)

    if animation_mode == 'Video Input':
        frames = sorted(glob(in_path + '/*.*'));
        if len(frames) == 0:
            sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")
        flows = glob(flo_folder + '/*.*')
        if (len(flows) == 0) and flow_warp:
            sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")
    settings_out = batchFolder + f"/settings"
    if resume_run:
        if run_to_resume == 'latest':
            try:
                batchNum
            except:
                batchNum = len(glob(f"{settings_out}/{batch_name}(*)_settings.txt")) - 1
        else:
            batchNum = int(run_to_resume)
        if resume_from_frame == 'latest':
            start_frame = len(glob(batchFolder + f"/{batch_name}({batchNum})_*.png"))
            if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(turbo_steps))
        else:
            start_frame = int(resume_from_frame) + 1
            if animation_mode != 'Video Input' and turbo_mode == True and start_frame > turbo_preroll and start_frame % int(turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(turbo_steps))
            if retain_overwritten_frames is True:
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

    if set_seed == 'random_seed' or set_seed == -1:
        random.seed()
        seed = random.randint(0, 2 ** 32)
        # print(f'Using seed: {seed}')
    else:
        seed = int(set_seed)

    new_prompt_loras = {}
    if text_prompts:
        _, new_prompt_loras = split_lora_from_prompts(text_prompts)
        print('Inferred loras schedule:\n', new_prompt_loras)

    if new_prompt_loras not in [{}, [], '', None]:
        inject_lora(sd_model)
        # load_loras(use_loras,lora_multipliers)

    args = {
        'batchNum': batchNum,
        'prompts_series': text_prompts if text_prompts else None,
        'rec_prompts_series': rec_prompts if rec_prompts else None,
        'neg_prompts_series': negative_prompts if negative_prompts else None,
        'image_prompts_series': image_prompts if image_prompts else None,

        'seed': seed,
        'display_rate': display_rate,
        'n_batches': n_batches if animation_mode == 'None' else 1,
        'batch_size': batch_size,
        'batch_name': batch_name,
        'steps': steps,
        'diffusion_sampling_mode': diffusion_sampling_mode,
        'width_height': width_height,
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'sat_scale': sat_scale,
        'cutn_batches': cutn_batches,
        'init_image': init_image,
        'init_scale': init_scale,
        'skip_steps': skip_steps,
        'side_x': side_x,
        'side_y': side_y,
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
        'animation_mode': animation_mode,
        'video_init_path': video_init_path,
        'extract_nth_frame': extract_nth_frame,
        'video_init_seed_continuity': video_init_seed_continuity,
        'key_frames': key_frames,
        'max_frames': max_frames if animation_mode != "None" else 1,
        'interp_spline': interp_spline,
        'start_frame': start_frame,
        'padding_mode': padding_mode,
        'text_prompts': text_prompts,
        'image_prompts': image_prompts,
        'intermediate_saves': intermediate_saves,
        'intermediates_in_subfolder': intermediates_in_subfolder,
        'steps_per_checkpoint': steps_per_checkpoint,
        'perlin_init': perlin_init,
        'perlin_mode': perlin_mode,
        'set_seed': set_seed,
        'clamp_grad': clamp_grad,
        'clamp_max': clamp_max,
        'skip_augs': skip_augs,
    }
    if frame_range not in [None, [0, 0], '', [0], 0]:
        args['start_frame'] = frame_range[0]
        args['max_frames'] = min(args['max_frames'], frame_range[1])
    args = SimpleNamespace(**args)

    import traceback

    gc.collect()

    do_run()
    print('n_stats_avg (mean, std): ', n_mean_avg, n_std_avg)

    gc.collect()
    torch.cuda.empty_cache()
