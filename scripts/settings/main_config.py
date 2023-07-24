from dataclasses import dataclass

import torch
from PIL import Image

from k_diffusion.sampling import sample_euler
from python_color_transfer.color_transfer import ColorTransfer
from scripts.settings.setting import steps
from scripts.utils.env import root_dir

@dataclass
class MainConfig:
    # @dataclass
    # class NoGuiConfig:
    # @title Flow and turbo settings
    # cal optical flow from video frames and warp prev frame with flow
    flow_blend = 0.999
    ##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame
    check_consistency = True  # @param {type: 'boolean'}
    # cal optical flow from video frames and warp prev frame with flow

    # ======= TURBO MODE
    # @markdown ---
    # @markdown ####**Turbo Mode:**
    # @markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
    # @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
    # @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

    turbo_mode = False  # @param {type:"boolean"}
    turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
    turbo_preroll = 1  # frames
    # @title Consistency map mixing
    # @markdown You can mix consistency map layers separately\
    # @markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
    # @markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
    # @markdown edges_consistency_weight - masks moving objects' edges\
    # @markdown The default values to simulate previous versions' behavior are 1,1,1

    missed_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
    overshoot_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
    edges_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
    # @title  ####**Seed and grad Settings:**

    set_seed = '4275770367'  # @param{type: 'string'}

    # @markdown *Clamp grad is used with any of the init_scales or sat_scale above 0*\
    # @markdown Clamp grad limits the amount various criterions, controlled by *_scale parameters, are pushing the image towards the desired result.\
    # @markdown For example, high scale values may cause artifacts, and clamp_grad removes this effect.
    # @markdown 0.7 is a good clamp_max value.
    eta = 0.55
    clamp_grad = True  # @param{type: 'boolean'}
    clamp_max = 2  # @param{type: 'number'}
    text_prompts = {0: ['Masterpiece, beautiful white marble statue,sculpture,white hair']}

    negative_prompts = {
        0: ["text, naked, nude, logo, cropped, two heads, four arms, lazy eye, blurry, unfocused"]
    }
    # @title ##Warp Turbo Smooth Settings
    # @markdown Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.
    turbo_frame_skips_steps = '100% (don`t diffuse turbo frames, fastest)'  # @param ['70%','75%','80%','85%', '90%', '95%', '100% (don`t diffuse turbo frames, fastest)']

    if turbo_frame_skips_steps == '100% (don`t diffuse turbo frames, fastest)':
        turbo_frame_skips_steps = None
    else:
        turbo_frame_skips_steps = int(turbo_frame_skips_steps.split('%')[0]) / 100
    # None - disable and use default skip steps

    # @markdown ###Consistency mask postprocessing
    # @markdown ####Soften consistency mask
    # @markdown Lower values mean less stylized frames and more raw video input in areas with fast movement, but fewer trails add ghosting.\
    # @markdown Gives glitchy datamoshing look.\
    # @markdown Higher values keep stylized frames, but add trails and ghosting.

    soften_consistency_mask = 0  # @param {type:"slider", min:0, max:1, step:0.1}
    forward_weights_clip = soften_consistency_mask
    # 0 behaves like consistency on, 1 - off, in between - blends
    soften_consistency_mask_for_turbo_frames = 0  # @param {type:"slider", min:0, max:1, step:0.1}
    forward_weights_clip_turbo_step = soften_consistency_mask_for_turbo_frames
    # None - disable and use forward_weights_clip for turbo frames, 0 behaves like consistency on, 1 - off, in between - blends
    # @markdown ####Blur consistency mask.
    # @markdown Softens transition between raw video init and stylized frames in occluded areas.
    consistency_blur = 1  # @param
    # @markdown ####Dilate consistency mask.
    # @markdown Expands consistency mask without blurring the edges.
    consistency_dilate = 3  # @param

    disable_cc_for_turbo_frames = False  # @param {"type":"boolean"}
    # disable consistency for turbo frames, the same as forward_weights_clip_turbo_step = 1, but a bit faster

    # @markdown ###Frame padding
    # @markdown Increase padding if you have a shaky\moving camera footage and are getting black borders.

    padding_ratio = 0.2  # @param {type:"slider", min:0, max:1, step:0.1}
    # relative to image size, in range 0-1
    padding_mode = 'reflect'  # @param ['reflect','edge','wrap']

    # safeguard the params
    if turbo_frame_skips_steps is not None:
        turbo_frame_skips_steps = min(max(0, turbo_frame_skips_steps), 1)
    forward_weights_clip = min(max(0, forward_weights_clip), 1)
    if forward_weights_clip_turbo_step is not None:
        forward_weights_clip_turbo_step = min(max(0, forward_weights_clip_turbo_step), 1)
    padding_ratio = min(max(0, padding_ratio), 1)
    ##@markdown ###Inpainting
    ##@markdown Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.

    inpaint_blend = 0
    ##@param {type:"slider", min:0,max:1,value:1,step:0.1}

    # @markdown ###Color matching
    # @markdown Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted\
    # @markdown 0 - off, other values control effect opacity

    match_color_strength = 0  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

    disable_cc_for_turbo_frames = False
    # @title Video mask settings
    # @markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
    use_background_mask = False  # @param {'type':'boolean'}
    # @markdown Check to invert the mask.
    invert_mask = False  # @param {'type':'boolean'}
    # @markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
    apply_mask_after_warp = True  # @param {'type':'boolean'}
    # @markdown Choose background source to paste masked stylized image onto: image, color, init video.
    background = "init_video"  # @param ['image', 'color', 'init_video']
    # @markdown Specify the init image path or color depending on your background source choice.
    background_source = 'red'  # @param {'type':'string'}

    # @title Frame correction
    # @markdown Match frame pixels or latent to other frames to preven oversaturation and feedback loop artifacts
    # @markdown ###Latent matching
    # @markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
    normalize_latent = 'off'  # @param ['off', 'color_video', 'color_video_offset', 'user_defined', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
    # @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.

    normalize_latent_offset = 0  # @param {'type':'number'}
    # @markdown User defined stats to normalize the latent towards
    latent_fixed_mean = 0.  # @param {'type':'raw'}
    latent_fixed_std = 0.9  # @param {'type':'raw'}
    # @markdown Match latent on per-channel basis
    latent_norm_4d = True  # @param {'type':'boolean'}
    # @markdown ###Color matching
    # @markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
    colormatch_frame = 'off'  # @param ['off', 'color_video', 'color_video_offset','stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
    # @markdown Color match strength. 1 mimics legacy behavior
    color_match_frame_str = 0.5  # @param {'type':'number'}
    # @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.
    colormatch_offset = 0  # @param {'type':'number'}
    colormatch_method = 'PDF'  # @param ['LAB', 'PDF', 'mean']
    colormatch_method_fn = ColorTransfer.lab_transfer
    if colormatch_method == 'LAB':
        colormatch_method_fn = ColorTransfer.pdf_transfer
    if colormatch_method == 'mean':
        colormatch_method_fn = ColorTransfer.mean_std_transfer
    # @markdown Match source frame's texture
    colormatch_regrain = False  # @param {'type':'boolean'}
    warp_mode = 'use_image'  # @param ['use_latent', 'use_image']
    warp_towards_init = 'off'  # @param ['stylized', 'off']
    # if warp_towards_init != 'off':
    #     if flow_lq:
    #         raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
    #     else:
    #         raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()

    cond_image_src = 'init'  # @param ['init', 'stylized']
    # DD-style losses, renders 2 times slower (!) and more memory intensive :D

    latent_scale_schedule = [0,
                             0]  # controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
    init_scale_schedule = [0, 0]  # controls coherency with previous frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
    sat_scale = 0

    init_grad = False  # True - compare result to real frame, False - to stylized frame
    grad_denoised = True  # fastest, on by default, calc grad towards denoised x instead of input x
    steps_schedule = {
        0: 25
    }  # schedules total steps. useful with low strength, when you end up with only 10 steps at 0.2 strength x50 steps. Increasing max steps for low strength gives model more time to get to your text prompt
    style_strength_schedule = [
        0.7]  # [0.5]+[0.2]*149+[0.3]*3+[0.2] #use this instead of skip steps. It means how many steps we should do. 0.8 = we diffuse for 80% steps, so we skip 20%. So for skip steps 70% use 0.3
    flow_blend_schedule = [0.8]  # for example [0.1]*3+[0.999]*18+[0.3] will fade-in for 3 frames, keep style for 18 frames, and fade-out for the rest
    cfg_scale_schedule = [15]  # text2image strength, 7.5 is a good default
    blend_json_schedules = True  # True - interpolate values between keyframes. False - use latest keyframe

    dynamic_thresh = 30

    fixed_code = False  # Aka fixed seed. you can use this with fast moving videos, but be careful with still images
    code_randomness = 0.1  # Only affects fixed code. high values make the output collapse
    # normalize_code = True #Only affects fixed code.

    warp_strength = 1  # leave 1 for no change. 1.01 is already a strong value.
    flow_override_map = []  # [*range(1,15)]+[16]*10+[*range(17+10,17+10+20)]+[18+10+20]*15+[*range(19+10+20+15,9999)] #map flow to frames. set to [] to disable.  [1]*10+[*range(10,9999)] repeats 1st frame flow 10 times, then continues as usual

    blend_latent_to_init = 0

    colormatch_after = False  # colormatch after stylizing. On in previous notebooks.
    colormatch_turbo = False  # apply colormatching for turbo frames. On in previous notebooks

    user_comment = 'testing cc layers'

    mask_result = False  # imitates inpainting by leaving only inconsistent areas to be diffused

    use_karras_noise = False  # Should work better with current sample, needs more testing.
    end_karras_ramp_early = False

    warp_interp = Image.LANCZOS
    VERBOSE = True

    use_patchmatch_inpaiting = 0

    warp_num_k = 128  # number of patches per frame
    warp_forward = False  # use k-means patched warping (moves large areas instead of single pixels)

    inverse_inpainting_mask = False
    inpainting_mask_weight = 1.
    mask_source = 'none'
    mask_clip = [0, 255]
    sampler = sample_euler
    image_scale = 2
    image_scale_schedule = {0: 1.5, 1: 2}

    inpainting_mask_source = 'none'

    fixed_seed = False  # fixes seed
    offload_model = True  # offloads model to cpu defore running decoder. May save a bit of VRAM

    use_predicted_noise = False
    rec_randomness = 0.
    rec_cfg = 1.
    rec_prompts = {0: ['woman walking on a treadmill']}
    rec_source = 'init'
    rec_steps_pct = 1

    # controlnet settings
    controlnet_preprocess = True  # preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing
    detect_resolution = 768  # control net conditioning image resolution
    bg_threshold = 0.4  # controlnet depth/normal bg cutoff threshold
    low_threshold = 100  # canny filter parameters
    high_threshold = 200  # canny filter parameters
    value_threshold = 0.1  # mlsd model settings
    distance_threshold = 0.1  # mlsd model settings

    temporalnet_source = 'stylized'
    temporalnet_skip_1st_frame = True
    controlnet_multimodel_mode = 'internal'  # external or internal. internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.)

    do_softcap = False  # softly clamp latent excessive values. reduces feedback loop effect a bit
    softcap_thresh = 0.9  # scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected)
    softcap_q = 1.  # percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%)

    max_faces = 10
    masked_guidance = False  # use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas
    cc_masked_diffusion = 0.5  # 0 - off. 0.5-0.7 are good values. make inconsistent area passes only before this % of actual steps, then diffuse whole image
    alpha_masked_diffusion = 0.5  # 0 - off. 0.5-0.7 are good values. make alpha masked area passes only before this % of actual steps, then diffuse whole image
    invert_alpha_masked_diffusion = False

    save_controlnet_annotations = True
    control_sd15_openpose_hands_face = True
    control_sd15_depth_detector = 'Zoe'  # Zoe or Midas
    control_sd15_softedge_detector = 'PIDI'  # HED or PIDI
    control_sd15_seg_detector = 'Seg_UFADE20K'  # Seg_OFCOCO Seg_OFADE20K Seg_UFADE20K
    control_sd15_scribble_detector = 'PIDI'  # HED or PIDI
    control_sd15_lineart_coarse = False
    control_sd15_inpaint_mask_source = 'consistency_mask'  # consistency_mask, None, cond_video
    control_sd15_shuffle_source = 'color_video'  # color_video, init, prev_frame, first_frame
    control_sd15_shuffle_1st_source = 'color_video'  # color_video, init, None,
    overwrite_rec_noise = False

    controlnet_multimodel = {
        "control_sd15_depth": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_canny": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_softedge": {
            "weight": 1,
            "start": 0,
            "end": 1
        },
        "control_sd15_mlsd": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_normalbae": {
            "weight": 1,
            "start": 0,
            "end": 1
        },
        "control_sd15_openpose": {
            "weight": 1,
            "start": 0,
            "end": 1
        },
        "control_sd15_scribble": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_seg": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_temporalnet": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_face": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_ip2p": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_inpaint": {
            "weight": 1,
            "start": 0,
            "end": 1
        },
        "control_sd15_lineart": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_lineart_anime": {
            "weight": 0,
            "start": 0,
            "end": 1
        },
        "control_sd15_shuffle": {
            "weight": 0,
            "start": 0,
            "end": 1
        }
    }
    # these variables are not in the GUI and are not being loaded.

    # torch.backends.cudnn.enabled = True # disabling this may increase performance on Ampere and Ada GPUs

    diffuse_inpaint_mask_blur = 25  # used in mask result to extent the mask
    diffuse_inpaint_mask_thresh = 0.8  # used in mask result to extent the mask

    add_noise_to_latent = True  # add noise to latent vector during latent guidance
    noise_upscale_ratio = 1  # noise upscale ratio for latent noise during latent guidance
    guidance_use_start_code = True  # fix latent noise across steps during latent guidance
    use_scale = False  # use gradient scaling (for mixed precision)
    g_invert_mask = False  # invert guidance mask

    cb_noise_upscale_ratio = 1  # noise in masked diffusion callback
    cb_add_noise_to_latent = True  # noise in masked diffusion callback
    cb_use_start_code = True  # fix noise per frame in masked diffusion callback
    cb_fixed_code = False  # fix noise across all animation in masked diffusion callback (overcooks fast af)
    cb_norm_latent = False  # norm cb latent to normal ditribution stats in masked diffusion callback


    use_legacy_fixed_code = False

    deflicker_scale = 0.
    deflicker_latent_scale = 0

    prompt_patterns_sched = {}

    # run private config
    start_code_cb = None  # variable for cb_code
    guidance_start_code = None  # variable for guidance code

    display_size = 512  # @param
    image_prompts = {}
    intermediate_saves = None
    intermediates_in_subfolder = True
    steps_per_checkpoint = None
    max_frames = 0

    reference_latent = None

    key_frames = True
    interp_spline = 'Linear'
    perlin_init = False
    perlin_mode = 'mixed'

    ##@markdown `n_batches` ignored with animation modes.
    display_rate = 9999999
    ##@param{type: 'number'}
    n_batches = 1
    ##@param{type: 'number'}
    start_code = None
    first_latent = None
    first_latent_source = 'not set'
    n_mean_avg = None
    n_std_avg = None
    n_smooth = 0.5
    # Update Model Settings
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    inverse_mask_order = None
    batch_size = 1
    new_prompt_loras = {}
    # @markdown ---
    # @markdown Frames to run. Leave empty or [0,0] to run all frames.
    frame_range = [0, 0]  # @param
    resume_run = False  # @param{type: 'boolean'}
    run_to_resume = 'latest'  # @param{type: 'string'}
    resume_from_frame = 'latest'  # @param{type: 'string'}
    retain_overwritteon_frames = False  # @param{type: 'boolean'}

    args = None

    apply_depth = None
    apply_canny = None
    apply_mlsd = None
    apply_hed = None
    apply_openpose = None
    apply_seg = None
    apply_normal = None
    apply_softedge = None
    apply_scribble = None
    apply_shuffle = None
    apply_lineart = None
    apply_lineart_anime = None
    loaded_controlnets = {}
