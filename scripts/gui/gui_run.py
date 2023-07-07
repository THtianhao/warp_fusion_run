# @title gui

from IPython import display
from ipywidgets import VBox, Tab, Accordion, Button, Output
from pandas._typing import FilePath

from scripts.gui.gui_config import GUIConfig
from scripts.gui.gui_env import gui_controlnet, gui_difficulty_dict, guis
from scripts.gui.gui_func import set_visibility, load_settings, add_labels_dict
from scripts.model_process.model_env import model_version

def gui_run(config: GUIConfig):
    for key in gui_difficulty_dict[config.gui_difficulty]:
        for gui in guis:
            set_visibility(key, 'hidden', gui)

    gui_diffusion_label, gui_controlnet_label, gui_warp_label, gui_consistency_label, gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_label = [add_labels_dict(o) for o in guis]

    cond_keys = ['latent_scale_schedule', 'init_scale_schedule', 'clamp_grad', 'clamp_max', 'init_grad', 'grad_denoised', 'masked_guidance']
    conditioning_w = Accordion([VBox([gui_diffusion_label[o] for o in cond_keys])])
    conditioning_w.set_title(0, 'External Conditioning...')

    seed_keys = ['set_seed', 'fixed_seed', 'fixed_code', 'code_randomness']
    seed_w = Accordion([VBox([gui_diffusion_label[o] for o in seed_keys])])
    seed_w.set_title(0, 'Seed...')

    rec_keys = ['use_predicted_noise', 'rec_prompts', 'rec_cfg', 'rec_randomness', 'rec_source', 'rec_steps_pct', 'overwrite_rec_noise']
    rec_w = Accordion([VBox([gui_diffusion_label[o] for o in rec_keys])])
    rec_w.set_title(0, 'Reconstructed noise...')

    prompt_keys = ['text_prompts', 'negative_prompts', 'prompt_patterns_sched',
                   'steps_schedule', 'style_strength_schedule',
                   'cfg_scale_schedule', 'blend_latent_to_init', 'dynamic_thresh',
                   'cond_image_src', 'cc_masked_diffusion', 'alpha_masked_diffusion', 'invert_alpha_masked_diffusion']
    if model_version == 'v1_instructpix2pix':
        prompt_keys.append('image_scale_schedule')
    if model_version == 'v1_inpainting':
        prompt_keys += ['inpainting_mask_source', 'inverse_inpainting_mask', 'inpainting_mask_weight']
    prompt_keys = [o for o in prompt_keys if o not in seed_keys + cond_keys]
    prompt_w = [gui_diffusion_label[o] for o in prompt_keys]

    gui_diffusion_list = [*prompt_w, gui_diffusion_label['sampler'],
                          gui_diffusion_label['use_karras_noise'], conditioning_w, seed_w, rec_w]

    control_annotator_keys = ['controlnet_preprocess', 'save_controlnet_annotations', 'detect_resolution', 'bg_threshold', 'low_threshold', 'high_threshold', 'value_threshold',
                              'distance_threshold', 'max_faces', 'control_sd15_openpose_hands_face', 'control_sd15_depth_detector', 'control_sd15_softedge_detector',
                              'control_sd15_seg_detector', 'control_sd15_scribble_detector', 'control_sd15_lineart_coarse', 'control_sd15_inpaint_mask_source',
                              'control_sd15_shuffle_source', 'control_sd15_shuffle_1st_source', 'temporalnet_source', 'temporalnet_skip_1st_frame', ]
    control_annotator_w = Accordion([VBox([gui_controlnet_label[o] for o in control_annotator_keys])])
    control_annotator_w.set_title(0, 'Controlnet annotator settings...')
    controlnet_model_w = Accordion([gui_controlnet['controlnet_multimodel']])
    controlnet_model_w.set_title(0, 'Controlnet models settings...')
    control_keys = ['controlnet_multimodel_mode']
    control_w = [gui_controlnet_label[o] for o in control_keys]
    gui_control_list = [controlnet_model_w, control_annotator_w, *control_w]

    # misc
    misc_keys = ["user_comment", "blend_json_schedules", "VERBOSE", "offload_model"]
    misc_w = [gui_misc_label[o] for o in misc_keys]

    softcap_keys = ['do_softcap', 'softcap_thresh', 'softcap_q']
    softcap_w = Accordion([VBox([gui_misc_label[o] for o in softcap_keys])])
    softcap_w.set_title(0, 'Softcap settings...')

    load_settings_btn = Button(description='Load settings')

    def btn_eventhandler(obj):
        load_settings(load_settings_path.value,)

    load_settings_btn.on_click(btn_eventhandler)
    load_settings_path = FilePath(placeholder='Please specify the path to the settings file to load.', description_tooltip='Please specify the path to the settings file to load.')
    settings_w = Accordion([VBox([load_settings_path, load_settings_btn])])
    settings_w.set_title(0, 'Load settings...')
    gui_misc_list = [*misc_w, softcap_w, settings_w]

    guis_labels_source = [gui_diffusion_list]
    guis_titles_source = ['diffusion']
    if 'control' in model_version:
        guis_labels_source += [gui_control_list]
        guis_titles_source += ['controlnet']

    guis_labels_source += [gui_warp_label, gui_consistency_label,
                           gui_turbo_label, gui_mask_label, gui_colormatch_label, gui_misc_list]
    guis_titles_source += ['warp', 'consistency', 'turbo', 'mask', 'colormatch', 'misc']

    guis_labels = [VBox([*o.values()]) if isinstance(o, dict) else VBox(o) for o in guis_labels_source]

    app = Tab(guis_labels)
    for i, title in enumerate(guis_titles_source):
        app.set_title(i, title)

    output = Output()
    if config.default_settings_path != '' and config.load_default_settings:
        load_settings(config.default_settings_path)

    display.display(app)
