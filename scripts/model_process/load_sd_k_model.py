#  specify path to your Stable Diffusion checkpoint (the "original" flavor)
# @title define SD + K functions, load model
import gc
import pathlib
import sys
import os

from scripts.model_process.CFGDenoiser import CFGDenoiser
from scripts.model_process.instruct_pix2_cfg_denoiser import InstructPix2PixCFGDenoiser
from scripts.model_process.mode_func import cldm_forward
from scripts.model_process.model_config import ModelConfig
from scripts.model_process.model_env import model_version, model_urls, control_model_urls, control_helpers, load_model_from_config, vae_ckpt, config_path, quantize
from scripts.utils.env import root_dir
from omegaconf import OmegaConf
import wget
import k_diffusion
import pickle

def load_sd_and_k_fusion(config: ModelConfig):
    if config.controlnet_models_dir.startswith('/content') or config.controlnet_models_dir == '':
        controlnet_models_dir = f"{root_dir}/ControlNet/models"
        print('You have a controlnet path set up for google drive, but we are not on Colab. Defaulting controlnet model path to ', controlnet_models_dir)
    os.makedirs(config.controlnet_models_dir, exist_ok=True)
    #  ---

    control_sd15_canny = False
    control_sd15_depth = False
    control_sd15_softedge = True
    control_sd15_mlsd = False
    control_sd15_normalbae = False
    control_sd15_openpose = False
    control_sd15_scribble = False
    control_sd15_seg = False
    control_sd15_temporalnet = False
    control_sd15_face = False

    if model_version == 'control_multi':
        control_versions = []
        if control_sd15_canny: control_versions += ['control_sd15_canny']
        if control_sd15_depth: control_versions += ['control_sd15_depth']
        if control_sd15_softedge: control_versions += ['control_sd15_softedge']
        if control_sd15_mlsd: control_versions += ['control_sd15_mlsd']
        if control_sd15_normalbae: control_versions += ['control_sd15_normalbae']
        if control_sd15_openpose: control_versions += ['control_sd15_openpose']
        if control_sd15_scribble: control_versions += ['control_sd15_scribble']
        if control_sd15_seg: control_versions += ['control_sd15_seg']
        if control_sd15_temporalnet: control_versions += ['control_sd15_temporalnet']
        if control_sd15_face: control_versions += ['control_sd15_face']
    else:
        control_versions = [model_version]

    if model_version in ["control_sd15_canny",
                         "control_sd15_depth",
                         "control_sd15_softedge",
                         "control_sd15_mlsd",
                         "control_sd15_normalbae",
                         "control_sd15_openpose",
                         "control_sd15_scribble",
                         "control_sd15_seg", 'control_sd15_face', 'control_multi']:

        os.chdir(f"{root_dir}/ControlNet/")

        os.chdir('../')

        # if download model is on and model path is not found, download full controlnet
        if config.download_control_model:
            if not os.path.exists(config.model_path):
                print(f'Model not found at {config.model_path}')
                if model_version == 'control_multi':
                    model_ver = control_versions[0]
                else:
                    model_ver = model_version

                model_path = f"{config.controlnet_models_dir}/v1-5-pruned-emaonly.safetensors"
                model_url = model_urls["sd_v1_5"]
                if not os.path.exists(model_path) or config.force_download:
                    try:
                        pathlib.Path(model_path).unlink()
                    except:
                        pass
                    print('Downloading full sd v1.5 model to ', model_path)
                    wget.download(model_url, model_path)
                    print('Downloaded full model.')
            # if model found, assume it's a working checkpoint, download small controlnet only:

            for model_ver in control_versions:
                small_url = control_model_urls[model_ver]
                local_filename = small_url.split('/')[-1]
                small_controlnet_model_path = f"{config.controlnet_models_dir}/{local_filename}"
                if config.use_small_controlnet and os.path.exists(config.model_path) and not os.path.exists(small_controlnet_model_path):
                    print(f'Model found at {config.model_path}. Small model not found at {small_controlnet_model_path}.')

                    if not os.path.exists(small_controlnet_model_path) or config.force_download:
                        try:
                            pathlib.Path(small_controlnet_model_path).unlink()
                        except:
                            pass
                        print(f'Downloading small controlnet model from {small_url}... ')
                        wget.download(small_url, small_controlnet_model_path)
                        print('Downloaded small controlnet model.')

                # https://huggingface.co/lllyasviel/Annotators/tree/main
                # https://huggingface.co/lllyasviel/Annotators/resolve/main/150_16_swin_l_oneformer_coco_100ep.pth
                helper_names = control_helpers[model_ver]
                if helper_names is not None:
                    if type(helper_names) == str: helper_names = [helper_names]
                    for helper_name in helper_names:
                        helper_model_url = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/' + helper_name
                        helper_model_path = f'{root_dir}/ControlNet/annotator/ckpts/' + helper_name
                        if not os.path.exists(helper_model_path) or config.force_download:
                            try:
                                pathlib.Path(helper_model_path).unlink()
                            except:
                                pass
                            wget.download(helper_model_url, helper_model_path)
        assert os.path.exists(config.model_path), f'Model not found at path: {config.model_path}. Please enter a valid path to the checkpoint file.'

        if os.path.exists(config.small_controlnet_model_path):
            smallpath = config.small_controlnet_model_path
        else:
            smallpath = None
        load_config = OmegaConf.load(f"{root_dir}/ControlNet/models/cldm_v15.yaml")
        sd_model = load_model_from_config(config=load_config,
                                          ckpt=config.model_path, vae_ckpt=vae_ckpt,  # controlnet=smallpath,
                                          verbose=True)

        # legacy
        # sd_model = create_model(f"{root_dir}/ControlNet/models/cldm_v15.yaml").cuda()
        # sd_model.load_state_dict(load_state_dict(model_path, location=load_to), strict=False)
        sd_model.cond_stage_model.half()
        sd_model.model.half()
        sd_model.control_model.half()
        sd_model.cuda()

        gc.collect()
    else:
        assert os.path.exists(config.model_path), f'Model not found at path: {config.model_path}. Please enter a valid path to the checkpoint file.'
        if config.model_path.endswith('.pkl'):
            with open(config.model_path, 'rb') as f:
                sd_model = pickle.load(f).cuda().eval()
        else:
            config = OmegaConf.load(config_path)
            sd_model = load_model_from_config(config, config.model_path, vae_ckpt=vae_ckpt, verbose=True).cuda()

    sys.path.append('./stablediffusion/')

    # sd_model.first_stage_model = torch.compile(sd_model.first_stage_model)
    # sd_model.model = torch.compile(sd_model.model)
    if sd_model.parameterization == "v":
        config.model_wrap = k_diffusion.external.CompVisVDenoiser(sd_model, quantize=quantize)
    else:
        config.model_wrap = k_diffusion.external.CompVisDenoiser(sd_model, quantize=quantize)
    config.sigma_min, config.sigma_max = config.model_wrap.sigmas[0].item(), config.model_wrap.sigmas[-1].item()
    config.model_wrap_cfg = CFGDenoiser(config.model_wrap)
    if model_version == 'v1_instructpix2pix':
        config.model_wrap_cfg = InstructPix2PixCFGDenoiser(config.model_wrap)
    try:
        sd_model.model.diffusion_model.forward = cldm_forward
    except Exception as e:
        print(e)
        # pass
    config.sd_mode = sd_model
