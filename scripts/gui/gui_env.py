# from ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, \
#     SelectionSlider, Valid
#
# from k_diffusion.sampling import sample_euler, sample_euler_ancestral, sample_heun, sample_dpm_2, sample_dpm_2_ancestral, sample_lms, sample_dpm_fast, sample_dpm_adaptive, sample_dpmpp_2s_ancestral, \
#     sample_dpmpp_sde, sample_dpmpp_2m
# from scripts.gui.control_gui import ControlGUI
# from scripts.gui.gui_func import get_value
# from scripts.gui.gui_run import guis
# from scripts.settings.main_settings import offload_model, mask_clip, rec_source
#
# # @markdown Load default settings
# gui_difficulty_dict = {
#     "I'm too young to die.": ["flow_warp", "warp_strength", "warp_mode", "padding_mode", "padding_ratio",
#                               "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k", "warp_forward",
#                               "blend_json_schedules", "VERBOSE", "offload_model", "do_softcap", "softcap_thresh",
#                               "softcap_q", "user_comment", "turbo_mode", "turbo_steps", "colormatch_turbo",
#                               "turbo_frame_skips_steps", "soften_consistency_mask_for_turbo_frames", "check_consistency",
#                               "missed_consistency_weight", "overshoot_consistency_weight", "edges_consistency_weight",
#                               "soften_consistency_mask", "consistency_blur", "match_color_strength", "mask_result",
#                               "use_patchmatch_inpaiting", "normalize_latent", "normalize_latent_offset", "latent_fixed_mean",
#                               "latent_fixed_std", "latent_norm_4d", "use_karras_noise", "cond_image_src", "inpainting_mask_source",
#                               "inverse_inpainting_mask", "inpainting_mask_weight", "init_grad", "grad_denoised",
#                               "image_scale_schedule", "blend_latent_to_init", "dynamic_thresh", "rec_cfg", "rec_source",
#                               "rec_steps_pct", "controlnet_multimodel_mode",
#                               "overwrite_rec_noise",
#                               "colormatch_after", "sat_scale", "clamp_grad", "apply_mask_after_warp"],
#     "Hey, not too rough.": ["flow_warp", "warp_strength", "warp_mode",
#                             "warp_towards_init", "flow_override_map", "mask_clip", "warp_num_k", "warp_forward",
#                             "check_consistency",
#                             "use_patchmatch_inpaiting", "init_grad", "grad_denoised",
#                             "image_scale_schedule", "blend_latent_to_init", "rec_cfg",
#                             "colormatch_after", "sat_scale", "clamp_grad", "apply_mask_after_warp"],
#     "Hurt me plenty.": "",
#     "Ultra-Violence.": []
# }
#
#
# # try keep settings on occasional run cell
# latent_scale_schedule = eval(get_value('latent_scale_schedule', guis))
# init_scale_schedule = eval(get_value('init_scale_schedule', guis))
# steps_schedule = eval(get_value('steps_schedule', guis))
# style_strength_schedule = eval(get_value('style_strength_schedule', guis))
# cfg_scale_schedule = eval(get_value('cfg_scale_schedule', guis))
# flow_blend_schedule = eval(get_value('flow_blend_schedule', guis))
# image_scale_schedule = eval(get_value('image_scale_schedule', guis))
#
# user_comment = get_value('user_comment', guis)
# blend_json_schedules = get_value('blend_json_schedules', guis)
# VERBOSE = get_value('VERBOSE', guis)
# use_background_mask = get_value('use_background_mask', guis)
# invert_mask = get_value('invert_mask', guis)
# background = get_value('background', guis)
# background_source = get_value('background_source', guis)
# (mask_clip_low, mask_clip_high) = get_value('mask_clip', guis)
#
# # turbo
# turbo_mode = get_value('turbo_mode', guis)
# turbo_steps = get_value('turbo_steps', guis)
# colormatch_turbo = get_value('colormatch_turbo', guis)
# turbo_frame_skips_steps = get_value('turbo_frame_skips_steps', guis)
# soften_consistency_mask_for_turbo_frames = get_value('soften_consistency_mask_for_turbo_frames', guis)
#
# # warp
# flow_warp = get_value('flow_warp', guis)
# apply_mask_after_warp = get_value('apply_mask_after_warp', guis)
# warp_num_k = get_value('warp_num_k', guis)
# warp_forward = get_value('warp_forward', guis)
# warp_strength = get_value('warp_strength', guis)
# flow_override_map = eval(get_value('flow_override_map', guis))
# warp_mode = get_value('warp_mode', guis)
# warp_towards_init = get_value('warp_towards_init', guis)
#
# # cc
# check_consistency = get_value('check_consistency', guis)
# missed_consistency_weight = get_value('missed_consistency_weight', guis)
# overshoot_consistency_weight = get_value('overshoot_consistency_weight', guis)
# edges_consistency_weight = get_value('edges_consistency_weight', guis)
# consistency_blur = get_value('consistency_blur', guis)
# consistency_dilate = get_value('consistency_dilate', guis)
# padding_ratio = get_value('padding_ratio', guis)
# padding_mode = get_value('padding_mode', guis)
# match_color_strength = get_value('match_color_strength', guis)
# soften_consistency_mask = get_value('soften_consistency_mask', guis)
# mask_result = get_value('mask_result', guis)
# use_patchmatch_inpaiting = get_value('use_patchmatch_inpaiting', guis)
#
# # diffusion
# text_prompts = eval(get_value('text_prompts', guis))
# negative_prompts = eval(get_value('negative_prompts', guis))
# prompt_patterns_sched = eval(get_value('prompt_patterns_sched', guis))
# cond_image_src = get_value('cond_image_src', guis)
# set_seed = get_value('set_seed', guis)
# clamp_grad = get_value('clamp_grad', guis)
# clamp_max = get_value('clamp_max', guis)
# sat_scale = get_value('sat_scale', guis)
# init_grad = get_value('init_grad', guis)
# grad_denoised = get_value('grad_denoised', guis)
# blend_latent_to_init = get_value('blend_latent_to_init', guis)
# fixed_code = get_value('fixed_code', guis)
# code_randomness = get_value('code_randomness', guis)
# # normalize_code=get_value('normalize_code',guis)
# dynamic_thresh = get_value('dynamic_thresh', guis)
# sampler = get_value('sampler', guis)
# use_karras_noise = get_value('use_karras_noise', guis)
# inpainting_mask_weight = get_value('inpainting_mask_weight', guis)
# inverse_inpainting_mask = get_value('inverse_inpainting_mask', guis)
# inpainting_mask_source = get_value('mask_source', guis)
#
# # colormatch
# normalize_latent = get_value('normalize_latent', guis)
# normalize_latent_offset = get_value('normalize_latent_offset', guis)
# latent_fixed_mean = eval(str(get_value('latent_fixed_mean', guis)))
# latent_fixed_std = eval(str(get_value('latent_fixed_std', guis)))
# latent_norm_4d = get_value('latent_norm_4d', guis)
# colormatch_frame = get_value('colormatch_frame', guis)
# color_match_frame_str = get_value('color_match_frame_str', guis)
# colormatch_offset = get_value('colormatch_offset', guis)
# colormatch_method = get_value('colormatch_method', guis)
# colormatch_regrain = get_value('colormatch_regrain', guis)
# colormatch_after = get_value('colormatch_after', guis)
# image_prompts = {}
#
# fixed_seed = get_value('fixed_seed', guis)
#
# # rec noise
# rec_cfg = get_value('rec_cfg', guis)
# rec_steps_pct = get_value('rec_steps_pct', guis)
# rec_prompts = eval(get_value('rec_prompts', guis))
# rec_randomness = get_value('rec_randomness', guis)
# use_predicted_noise = get_value('use_predicted_noise', guis)
# overwrite_rec_noise = get_value('overwrite_rec_noise', guis)
#
# # controlnet
# save_controlnet_annotations = get_value('save_controlnet_annotations', guis)
# control_sd15_openpose_hands_face = get_value('control_sd15_openpose_hands_face', guis)
# control_sd15_depth_detector = get_value('control_sd15_depth_detector', guis)
# control_sd15_softedge_detector = get_value('control_sd15_softedge_detector', guis)
# control_sd15_seg_detector = get_value('control_sd15_seg_detector', guis)
# control_sd15_scribble_detector = get_value('control_sd15_scribble_detector', guis)
# control_sd15_lineart_coarse = get_value('control_sd15_lineart_coarse', guis)
# control_sd15_inpaint_mask_source = get_value('control_sd15_inpaint_mask_source', guis)
# control_sd15_shuffle_source = get_value('control_sd15_shuffle_source', guis)
# control_sd15_shuffle_1st_source = get_value('control_sd15_shuffle_1st_source', guis)
# controlnet_multimodel = get_value('controlnet_multimodel', guis)
#
# controlnet_preprocess = get_value('controlnet_preprocess', guis)
# detect_resolution = get_value('detect_resolution', guis)
# bg_threshold = get_value('bg_threshold', guis)
# low_threshold = get_value('low_threshold', guis)
# high_threshold = get_value('high_threshold', guis)
# value_threshold = get_value('value_threshold', guis)
# distance_threshold = get_value('distance_threshold', guis)
# temporalnet_source = get_value('temporalnet_source', guis)
# temporalnet_skip_1st_frame = get_value('temporalnet_skip_1st_frame', guis)
# controlnet_multimodel_mode = get_value('controlnet_multimodel_mode', guis)
# max_faces = get_value('max_faces', guis)
#
# do_softcap = get_value('do_softcap', guis)
# softcap_thresh = get_value('softcap_thresh', guis)
# softcap_q = get_value('softcap_q', guis)
#
# masked_guidance = get_value('masked_guidance', guis)
# cc_masked_diffusion = get_value('cc_masked_diffusion', guis)
# alpha_masked_diffusion = get_value('alpha_masked_diffusion', guis)
# invert_alpha_masked_diffusion = get_value('invert_alpha_masked_diffusion', guis)
#
#
# gui_misc = {
#     "user_comment": Textarea(value=user_comment, layout=Layout(width=f'80%'), description='user_comment:', description_tooltip='Enter a comment to differentiate between save files.'),
#     "blend_json_schedules": Checkbox(value=blend_json_schedules, description='blend_json_schedules', indent=True, description_tooltip='Smooth values between keyframes.',
#                                      tooltip='Smooth values between keyframes.'),
#     "VERBOSE": Checkbox(value=VERBOSE, description='VERBOSE', indent=True, description_tooltip='Print all logs'),
#     "offload_model": Checkbox(value=offload_model, description='offload_model', indent=True, description_tooltip='Offload unused models to CPU and back to GPU to save VRAM. May reduce speed.'),
#     "do_softcap": Checkbox(value=do_softcap, description='do_softcap', indent=True, description_tooltip='Softly clamp latent excessive values. Reduces feedback loop effect a bit.'),
#     "softcap_thresh": FloatSlider(value=softcap_thresh, min=0, max=1, step=0.05, description='softcap_thresh:', readout=True, readout_format='.1f',
#                                   description_tooltip='Scale down absolute values above that threshold (latents are being clamped at [-1:1] range, so 0.9 will downscale values above 0.9 to fit into that range, [-1.5:1.5] will be scaled to [-1:1], but only absolute values over 0.9 will be affected'),
#     "softcap_q": FloatSlider(value=softcap_q, min=0, max=1, step=0.05, description='softcap_q:', readout=True, readout_format='.1f',
#                              description_tooltip='Percentile to downscale. 1-downscle full range with outliers, 0.9 - downscale only 90%  values above thresh, clamp 10%'),
#
# }
#
# gui_mask = {
#     "use_background_mask": Checkbox(value=use_background_mask, description='use_background_mask', indent=True,
#                                     description_tooltip='Enable masking. In order to use it, you have to either extract or provide an existing mask in Video Masking cell.\n'),
#     "invert_mask": Checkbox(value=invert_mask, description='invert_mask', indent=True,
#                             description_tooltip='Inverts the mask, allowing to process either backgroung or characters, depending on your mask.'),
#     "background": Dropdown(description='background',
#                            options=['image', 'color', 'init_video'], value=background,
#                            description_tooltip='Background type. Image - uses static image specified in background_source, color - uses fixed color specified in background_source, init_video - uses raw init video for masked areas.'),
#     "background_source": Text(value=background_source, description='background_source', description_tooltip='Specify image path or color name of hash.'),
#     "apply_mask_after_warp": Checkbox(value=apply_mask_after_warp, description='apply_mask_after_warp', indent=True,
#                                       description_tooltip='On to reduce ghosting. Apply mask after warping and blending warped image with current raw frame. If off, only current frame will be masked, previous frame will be warped and blended wuth masked current frame.'),
#     "mask_clip": IntRangeSlider(
#         value=mask_clip,
#         min=0,
#         max=255,
#         step=1,
#         description='Mask clipping:',
#         description_tooltip='Values below the selected range will be treated as black mask, values above - as white.',
#         disabled=False,
#         continuous_update=False,
#         orientation='horizontal',
#         readout=True)
#
# }
#
# gui_turbo = {
#     "turbo_mode": Checkbox(value=turbo_mode, description='turbo_mode', indent=True,
#                            description_tooltip='Turbo mode skips diffusion process on turbo_steps number of frames. Frames are still being warped and blended. Speeds up the render at the cost of possible trails an ghosting.'),
#     "turbo_steps": IntText(value=turbo_steps, description='turbo_steps:', description_tooltip='Number of turbo frames'),
#     "colormatch_turbo": Checkbox(value=colormatch_turbo, description='colormatch_turbo', indent=True,
#                                  description_tooltip='Apply frame color matching during turbo frames. May increease rendering speed, but may add minor flickering.'),
#     "turbo_frame_skips_steps": SelectionSlider(description='turbo_frame_skips_steps',
#                                                options=['70%', '75%', '80%', '85%', '80%', '95%', '100% (don`t diffuse turbo frames, fastest)'], value='100% (don`t diffuse turbo frames, fastest)',
#                                                description_tooltip='Skip steps for turbo frames. Select 100% to skip diffusion rendering for turbo frames completely.'),
#     "soften_consistency_mask_for_turbo_frames": FloatSlider(value=soften_consistency_mask_for_turbo_frames, min=0, max=1, step=0.05, description='soften_consistency_mask_for_turbo_frames:',
#                                                             readout=True, readout_format='.1f', description_tooltip='Clips the consistency mask, reducing it`s effect'),
#
# }
#
# gui_warp = {
#     "flow_warp": Checkbox(value=flow_warp, description='flow_warp', indent=True,
#                           description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
#
#     "flow_blend_schedule": Textarea(value=str(flow_blend_schedule), layout=Layout(width=f'80%'), description='flow_blend_schedule:',
#                                     description_tooltip='Blend current raw init video frame with previously stylised frame with respect to consistency mask. 0 - raw frame, 1 - stylized frame'),
#     "warp_num_k": IntText(value=warp_num_k, description='warp_num_k:',
#                           description_tooltip='Nubmer of clusters in forward-warp mode. The more - the smoother is the motion. Lower values move larger chunks of image at a time.'),
#     "warp_forward": Checkbox(value=warp_forward, description='warp_forward', indent=True,
#                              description_tooltip='Experimental. Enable patch-based flow warping. Groups pixels by motion direction and moves them together, instead of moving individual pixels.'),
#     # "warp_interp": Textarea(value='Image.LANCZOS',layout=Layout(width=f'80%'),  description = 'warp_interp:'),
#     "warp_strength": FloatText(value=warp_strength, description='warp_strength:', description_tooltip='Experimental. Motion vector multiplier. Provides a glitchy effect.'),
#     "flow_override_map": Textarea(value=str(flow_override_map), layout=Layout(width=f'80%'), description='flow_override_map:',
#                                   description_tooltip='Experimental. Motion vector maps mixer. Allows changing frame-motion vetor indexes or repeating motion, provides a glitchy effect.'),
#     "warp_mode": Dropdown(description='warp_mode', options=['use_latent', 'use_image'],
#                           value=warp_mode, description_tooltip='Experimental. Apply warp to latent vector. May get really blurry, but reduces feedback loop effect for slow movement'),
#     "warp_towards_init": Dropdown(description='warp_towards_init',
#                                   options=['stylized', 'off'], value=warp_towards_init,
#                                   description_tooltip='Experimental. After a frame is stylized, computes the difference between output and input for that frame, and warps the output back to input, preserving its shape.'),
#     "padding_ratio": FloatSlider(value=padding_ratio, min=0, max=1, step=0.05, description='padding_ratio:', readout=True, readout_format='.1f',
#                                  description_tooltip='Amount of padding. Padding is used to avoid black edges when the camera is moving out of the frame.'),
#     "padding_mode": Dropdown(description='padding_mode', options=['reflect', 'edge', 'wrap'],
#                              value=padding_mode),
# }
#
# # warp_interp = Image.LANCZOS
#
# gui_consistency = {
#     "check_consistency": Checkbox(value=check_consistency, description='check_consistency', indent=True,
#                                   description_tooltip='Enables consistency checking (CC). CC is used to avoid ghosting and trails, that appear due to lack of information while warping frames. It allows replacing motion edges, frame borders, incorrectly moved areas with raw init frame data.'),
#     "missed_consistency_weight": FloatSlider(value=missed_consistency_weight, min=0, max=1, step=0.05, description='missed_consistency_weight:', readout=True, readout_format='.1f',
#                                              description_tooltip='Multiplier for incorrectly predicted\moved areas. For example, if an object moves and background appears behind it. We can predict what to put in that spot, so we can either duplicate the object, resulting in trail, or use init video data for that region.'),
#     "overshoot_consistency_weight": FloatSlider(value=overshoot_consistency_weight, min=0, max=1, step=0.05, description='overshoot_consistency_weight:', readout=True, readout_format='.1f',
#                                                 description_tooltip='Multiplier for areas that appeared out of the frame. We can either leave them black or use raw init video.'),
#     "edges_consistency_weight": FloatSlider(value=edges_consistency_weight, min=0, max=1, step=0.05, description='edges_consistency_weight:', readout=True, readout_format='.1f',
#                                             description_tooltip='Multiplier for motion edges. Moving objects are most likely to leave trails, this option together with missed consistency weight helps prevent that, but in a more subtle manner.'),
#     "soften_consistency_mask": FloatSlider(value=soften_consistency_mask, min=0, max=1, step=0.05, description='soften_consistency_mask:', readout=True, readout_format='.1f'),
#     "consistency_blur": FloatText(value=consistency_blur, description='consistency_blur:'),
#     "consistency_dilate": FloatText(value=consistency_dilate, description='consistency_dilate:', description_tooltip='expand consistency mask without blurring the edges'),
#     "barely used": Label(' '),
#     "match_color_strength": FloatSlider(value=match_color_strength, min=0, max=1, step=0.05, description='match_color_strength:', readout=True, readout_format='.1f',
#                                         description_tooltip='Enables colormathing raw init video pixls in inconsistent areas only to the stylized frame. May reduce flickering for inconsistent areas.'),
#     "mask_result": Checkbox(value=mask_result, description='mask_result', indent=True, description_tooltip='Stylizes only inconsistent areas. Takes consistent areas from the previous frame.'),
#     "use_patchmatch_inpaiting": FloatSlider(value=use_patchmatch_inpaiting, min=0, max=1, step=0.05, description='use_patchmatch_inpaiting:', readout=True, readout_format='.1f',
#                                             description_tooltip='Uses patchmatch inapinting for inconsistent areas. Is slow.'),
# }
#
# gui_diffusion = {
#     "use_karras_noise": Checkbox(value=use_karras_noise, description='use_karras_noise', indent=True, description_tooltip='Enable for samplers that have K at their name`s end.'),
#     "sampler": Dropdown(description='sampler', options=[('sample_euler', sample_euler),
#                                                         ('sample_euler_ancestral', sample_euler_ancestral),
#                                                         ('sample_heun', sample_heun),
#                                                         ('sample_dpm_2', sample_dpm_2),
#                                                         ('sample_dpm_2_ancestral', sample_dpm_2_ancestral),
#                                                         ('sample_lms', sample_lms),
#                                                         ('sample_dpm_fast', sample_dpm_fast),
#                                                         ('sample_dpm_adaptive', sample_dpm_adaptive),
#                                                         ('sample_dpmpp_2s_ancestral', sample_dpmpp_2s_ancestral),
#                                                         ('sample_dpmpp_sde', sample_dpmpp_sde),
#                                                         ('sample_dpmpp_2m', sample_dpmpp_2m)], value=sampler),
#     "prompt_patterns_sched": Textarea(value=str(prompt_patterns_sched), layout=Layout(width=f'80%'), description='Replace patterns:'),
#     "text_prompts": Textarea(value=str(text_prompts), layout=Layout(width=f'80%'), description='Prompt:'),
#     "negative_prompts": Textarea(value=str(negative_prompts), layout=Layout(width=f'80%'), description='Negative Prompt:'),
#     "cond_image_src": Dropdown(description='cond_image_src', options=['init', 'stylized', 'cond_video'],
#                                value=cond_image_src, description_tooltip='Depth map source for depth model. It can either take raw init video frame or previously stylized frame.'),
#     "inpainting_mask_source": Dropdown(description='inpainting_mask_source', options=['none', 'consistency_mask', 'cond_video'],
#                                        value=inpainting_mask_source,
#                                        description_tooltip='Inpainting model mask source. none - full white mask (inpaint whole image), consistency_mask - inpaint inconsistent areas only'),
#     "inverse_inpainting_mask": Checkbox(value=inverse_inpainting_mask, description='inverse_inpainting_mask', indent=True, description_tooltip='Inverse inpainting mask'),
#     "inpainting_mask_weight": FloatSlider(value=inpainting_mask_weight, min=0, max=1, step=0.05, description='inpainting_mask_weight:', readout=True, readout_format='.1f',
#                                           description_tooltip='Inpainting mask weight. 0 - Disables inpainting mask.'),
#     "set_seed": IntText(value=set_seed, description='set_seed:', description_tooltip='Seed. Use -1 for random.'),
#     "clamp_grad": Checkbox(value=clamp_grad, description='clamp_grad', indent=True, description_tooltip='Enable limiting the effect of external conditioning per diffusion step'),
#     "clamp_max": FloatText(value=clamp_max, description='clamp_max:', description_tooltip='limit the effect of external conditioning per diffusion step'),
#     "latent_scale_schedule": Textarea(value=str(latent_scale_schedule), layout=Layout(width=f'80%'), description='latent_scale_schedule:',
#                                       description_tooltip='Latents scale defines how much minimize difference between output and input stylized image in latent space.'),
#     "init_scale_schedule": Textarea(value=str(init_scale_schedule), layout=Layout(width=f'80%'), description='init_scale_schedule:',
#                                     description_tooltip='Init scale defines how much minimize difference between output and input stylized image in RGB space.'),
#     "sat_scale": FloatText(value=sat_scale, description='sat_scale:', description_tooltip='Saturation scale limits oversaturation.'),
#     "init_grad": Checkbox(value=init_grad, description='init_grad', indent=True, description_tooltip='On - compare output to real frame, Off - to stylized frame'),
#     "grad_denoised": Checkbox(value=grad_denoised, description='grad_denoised', indent=True,
#                               description_tooltip='Fastest, On by default, calculate gradients with respect to denoised image instead of input image per diffusion step.'),
#     "steps_schedule": Textarea(value=str(steps_schedule), layout=Layout(width=f'80%'), description='steps_schedule:',
#                                description_tooltip='Total diffusion steps schedule. Use list format like [50,70], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:50, 20:70} format to specify keyframes only.'),
#     "style_strength_schedule": Textarea(value=str(style_strength_schedule), layout=Layout(width=f'80%'), description='style_strength_schedule:',
#                                         description_tooltip='Diffusion (style) strength. Actual number of diffusion steps taken (at 50 steps with 0.3 or 30% style strength you get 15 steps, which also means 35 0r 70% skipped steps). Inverse of skep steps. Use list format like [0.5,0.35], where each element corresponds to a frame, last element being repeated forever, or dictionary like {0:0.5, 20:0.35} format to specify keyframes only.'),
#     "cfg_scale_schedule": Textarea(value=str(cfg_scale_schedule), layout=Layout(width=f'80%'), description='cfg_scale_schedule:',
#                                    description_tooltip='Guidance towards text prompt. 7 is a good starting value, 1 is off (text prompt has no effect).'),
#     "image_scale_schedule": Textarea(value=str(image_scale_schedule), layout=Layout(width=f'80%'), description='image_scale_schedule:',
#                                      description_tooltip='Only used with InstructPix2Pix Model. Guidance towards text prompt. 1.5 is a good starting value'),
#     "blend_latent_to_init": FloatSlider(value=blend_latent_to_init, min=0, max=1, step=0.05, description='blend_latent_to_init:', readout=True, readout_format='.1f',
#                                         description_tooltip='Blend latent vector with raw init'),
#     # "use_karras_noise": Checkbox(value=False,description='use_karras_noise',indent=True),
#     # "end_karras_ramp_early": Checkbox(value=False,description='end_karras_ramp_early',indent=True),
#     "fixed_seed": Checkbox(value=fixed_seed, description='fixed_seed', indent=True, description_tooltip='Fixed seed.'),
#     "fixed_code": Checkbox(value=fixed_code, description='fixed_code', indent=True, description_tooltip='Fixed seed analog. Fixes diffusion noise.'),
#     "code_randomness": FloatSlider(value=code_randomness, min=0, max=1, step=0.05, description='code_randomness:', readout=True, readout_format='.1f',
#                                    description_tooltip='Fixed seed amount/effect strength.'),
#     # "normalize_code":Checkbox(value=normalize_code,description='normalize_code',indent=True, description_tooltip= 'Whether to normalize the noise after adding fixed seed.'),
#     "dynamic_thresh": FloatText(value=dynamic_thresh, description='dynamic_thresh:',
#                                 description_tooltip='Limit diffusion model prediction output. Lower values may introduce clamping/feedback effect'),
#     "use_predicted_noise": Checkbox(value=use_predicted_noise, description='use_predicted_noise', indent=True, description_tooltip='Reconstruct initial noise from init / stylized image.'),
#     "rec_prompts": Textarea(value=str(rec_prompts), layout=Layout(width=f'80%'), description='Rec Prompt:'),
#     "rec_randomness": FloatSlider(value=rec_randomness, min=0, max=1, step=0.05, description='rec_randomness:', readout=True, readout_format='.1f',
#                                   description_tooltip='Reconstructed noise randomness. 0 - reconstructed noise only. 1 - random noise.'),
#     "rec_cfg": FloatText(value=rec_cfg, description='rec_cfg:', description_tooltip='CFG scale for noise reconstruction. 1-1.9 are the best values.'),
#     "rec_source": Dropdown(description='rec_source', options=['init', 'stylized'],
#                            value=rec_source, description_tooltip='Source for noise reconstruction. Either raw init frame or stylized frame.'),
#     "rec_steps_pct": FloatSlider(value=rec_steps_pct, min=0, max=1, step=0.05, description='rec_steps_pct:', readout=True, readout_format='.2f',
#                                  description_tooltip='Reconstructed noise steps in relation to total steps. 1 = 100% steps.'),
#     "overwrite_rec_noise": Checkbox(value=overwrite_rec_noise, description='overwrite_rec_noise', indent=True,
#                                     description_tooltip='Overwrite reconstructed noise cache. By default reconstructed noise is not calculated if the settings haven`t changed too much. You can eit prompt, neg prompt, cfg scale,  style strength, steps withot reconstructing the noise every time.'),
#
#     "masked_guidance": Checkbox(value=masked_guidance, description='masked_guidance', indent=True,
#                                 description_tooltip='Use mask for init/latent guidance to ignore inconsistencies and only guide based on the consistent areas.'),
#     "cc_masked_diffusion": FloatSlider(value=cc_masked_diffusion, min=0, max=1, step=0.05,
#                                        description='cc_masked_diffusion:', readout=True, readout_format='.2f',
#                                        description_tooltip='0 - off. 0.5-0.7 are good values. Make inconsistent area passes only before this % of actual steps, then diffuse whole image.'),
#     "alpha_masked_diffusion": FloatSlider(value=alpha_masked_diffusion, min=0, max=1, step=0.05,
#                                           description='alpha_masked_diffusion:', readout=True, readout_format='.2f',
#                                           description_tooltip='0 - off. 0.5-0.7 are good values. Make alpha masked area passes only before this % of actual steps, then diffuse whole image.'),
#     "invert_alpha_masked_diffusion": Checkbox(value=invert_alpha_masked_diffusion, description='invert_alpha_masked_diffusion', indent=True,
#                                               description_tooltip='invert alpha ask for masked diffusion'),
#
# }
# gui_colormatch = {
#     "normalize_latent": Dropdown(description='normalize_latent',
#                                  options=['off', 'user_defined', 'color_video', 'color_video_offset',
#                                           'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'], value=normalize_latent,
#                                  description_tooltip='Normalize latent to prevent it from overflowing. User defined: use fixed input values (latent_fixed_*) Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset field below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset filed below).'),
#     "normalize_latent_offset": IntText(value=normalize_latent_offset, description='normalize_latent_offset:',
#                                        description_tooltip='Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
#     "latent_fixed_mean": FloatText(value=latent_fixed_mean, description='latent_fixed_mean:', description_tooltip='User defined mean value for normalize_latent=user_Defined mode'),
#     "latent_fixed_std": FloatText(value=latent_fixed_std, description='latent_fixed_std:', description_tooltip='User defined standard deviation value for normalize_latent=user_Defined mode'),
#     "latent_norm_4d": Checkbox(value=latent_norm_4d, description='latent_norm_4d', indent=True, description_tooltip='Normalize on a per-channel basis (on by default)'),
#     "colormatch_frame": Dropdown(description='colormatch_frame', options=['off', 'stylized_frame', 'color_video', 'color_video_offset', 'init_frame', 'stylized_frame_offset', 'init_frame_offset'],
#                                  value=colormatch_frame,
#                                  description_tooltip='Match frame colors to prevent it from overflowing.  Stylized/init frame - match towards stylized/init frame with a fixed number (specified in the offset filed below). Stylized\init frame offset - match to a frame with a number = current frame - offset (specified in the offset field below).'),
#     "color_match_frame_str": FloatText(value=color_match_frame_str, description='color_match_frame_str:', description_tooltip='Colormatching strength. 0 - no colormatching effect.'),
#     "colormatch_offset": IntText(value=colormatch_offset, description='colormatch_offset:',
#                                  description_tooltip='Offset from current frame number for *_frame_offset mode, or fixed frame number for *frame mode.'),
#     "colormatch_method": Dropdown(description='colormatch_method', options=['LAB', 'PDF', 'mean'], value=colormatch_method),
#     # "colormatch_regrain": Checkbox(value=False,description='colormatch_regrain',indent=True),
#     "colormatch_after": Checkbox(value=colormatch_after, description='colormatch_after', indent=True,
#                                  description_tooltip='On - Colormatch output frames when saving to disk, may differ from the preview. Off - colormatch before stylizing.'),
#
# }
#
# gui_controlnet = {
#     "controlnet_preprocess": Checkbox(value=controlnet_preprocess, description='controlnet_preprocess', indent=True,
#                                       description_tooltip='preprocess input conditioning image for controlnet. If false, use raw conditioning as input to the model without detection/preprocessing.'),
#     "detect_resolution": IntText(value=detect_resolution, description='detect_resolution:',
#                                  description_tooltip='Control net conditioning image resolution. The size of the image passed into controlnet preprocessors. Suggest keeping this as high as you can fit into your VRAM for more details.'),
#     "bg_threshold": FloatText(value=bg_threshold, description='bg_threshold:', description_tooltip='Control net depth/normal bg cutoff threshold'),
#     "low_threshold": IntText(value=low_threshold, description='low_threshold:', description_tooltip='Control net canny filter parameters'),
#     "high_threshold": IntText(value=high_threshold, description='high_threshold:', description_tooltip='Control net canny filter parameters'),
#     "value_threshold": FloatText(value=value_threshold, description='value_threshold:', description_tooltip='Control net mlsd filter parameters'),
#     "distance_threshold": FloatText(value=distance_threshold, description='distance_threshold:', description_tooltip='Control net mlsd filter parameters'),
#     "temporalnet_source": Dropdown(description='temporalnet_source', options=['init', 'stylized'],
#                                    value=temporalnet_source, description_tooltip='Temporalnet guidance source. Previous init or previous stylized frame'),
#     "temporalnet_skip_1st_frame": Checkbox(value=temporalnet_skip_1st_frame, description='temporalnet_skip_1st_frame', indent=True,
#                                            description_tooltip='Skip temporalnet for 1st frame (if not skipped, will use raw init for guidance'),
#     "controlnet_multimodel_mode": Dropdown(description='controlnet_multimodel_mode', options=['internal', 'external'], value=controlnet_multimodel_mode,
#                                            description_tooltip='internal - sums controlnet values before feeding those into diffusion model, external - sum outputs of differnet contolnets after passing through diffusion model. external seems slower but smoother.'),
#     "max_faces": IntText(value=max_faces, description='max_faces:', description_tooltip='Max faces to detect. Control net face parameters'),
#     "save_controlnet_annotations": Checkbox(value=save_controlnet_annotations, description='save_controlnet_annotations', indent=True,
#                                             description_tooltip='Save controlnet annotator predictions. They will be saved to your project dir /controlnetDebug folder.'),
#     "control_sd15_openpose_hands_face": Checkbox(value=control_sd15_openpose_hands_face, description='control_sd15_openpose_hands_face', indent=True,
#                                                  description_tooltip='Enable full openpose mode with hands and facial features.'),
#     "control_sd15_depth_detector": Dropdown(description='control_sd15_depth_detector', options=['Zoe', 'Midas'], value=control_sd15_depth_detector,
#                                             description_tooltip='Depth annotator model.'),
#     "control_sd15_softedge_detector": Dropdown(description='control_sd15_softedge_detector', options=['HED', 'PIDI'], value=control_sd15_softedge_detector,
#                                                description_tooltip='Softedge annotator model.'),
#     "control_sd15_seg_detector": Dropdown(description='control_sd15_seg_detector', options=['Seg_OFCOCO', 'Seg_OFADE20K', 'Seg_UFADE20K'], value=control_sd15_seg_detector,
#                                           description_tooltip='Segmentation annotator model.'),
#     "control_sd15_scribble_detector": Dropdown(description='control_sd15_scribble_detector', options=['HED', 'PIDI'], value=control_sd15_scribble_detector,
#                                                description_tooltip='Sccribble annotator model.'),
#     "control_sd15_lineart_coarse": Checkbox(value=control_sd15_lineart_coarse, description='control_sd15_lineart_coarse', indent=True,
#                                             description_tooltip='Coarse strokes mode.'),
#     "control_sd15_inpaint_mask_source": Dropdown(description='control_sd15_inpaint_mask_source', options=['consistency_mask', 'None', 'cond_video'], value=control_sd15_inpaint_mask_source,
#                                                  description_tooltip='Inpainting controlnet mask source. consistency_mask - inpaints inconsistent areas, None - whole image, cond_video - loads external mask'),
#     "control_sd15_shuffle_source": Dropdown(description='control_sd15_shuffle_source', options=['color_video', 'init', 'prev_frame', 'first_frame'], value=control_sd15_shuffle_source,
#                                             description_tooltip='Shuffle controlnet source. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (stylized+warped with consistency mask and flow_blend opacity), prev_frame - uses previously stylized frame (stylized, not warped), first_frame - first stylized frame'),
#     "control_sd15_shuffle_1st_source": Dropdown(description='control_sd15_shuffle_1st_source', options=['color_video', 'init', 'None'], value=control_sd15_shuffle_1st_source,
#                                                 description_tooltip='Set 1st frame source for shuffle model. If you need to geet the 1st frame style from your image, and for the consecutive frames you want to use the resulting stylized images. color_video: uses color video frames (or single image) as source, init - uses current frame`s init as source (raw video frame), None - skips this controlnet for the 1st frame. For example, if you like the 1st frame you`re getting and want to keep its style, but don`t want to use an external image as a source.'),
#     "controlnet_multimodel": ControlGUI(controlnet_multimodel)
# }
# guis = [gui_diffusion, gui_controlnet, gui_warp, gui_consistency, gui_turbo, gui_mask, gui_colormatch, gui_misc]
