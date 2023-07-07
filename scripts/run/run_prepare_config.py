# @title Do the Run!
# @markdown Preview max size
import gc
import os
import pathlib
import sys
from glob import glob

import torch
import wget
from safetensors import safe_open

from modules import shared, sd_hijack
# from scripts.gui.gui_env import guis, control_sd15_softedge_detector
from scripts.gui.gui_func import get_value
import copy

from scripts.model_process.model_config import ModelConfig
from scripts.model_process.model_env import model_version, control_model_urls, load_to
from scripts.settings.main_settings import controlnet_multimodel, control_sd15_depth_detector, control_sd15_softedge_detector, control_sd15_scribble_detector, control_sd15_seg_detector
from scripts.video_process.color_transfor_func import force_download

def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora

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
                    m, u = loaded_controlnets[controlnet].load_state_dict(sd, strict=True)
                    loaded_controlnets[controlnet].half()
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
