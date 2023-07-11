import os
from dataclasses import dataclass

import torch
import wget
from safetensors import safe_open

from ldm.util import instantiate_from_config
from scripts.utils.env import root_dir

dynamic_thresh = 2.
device = 'cuda'
# config_path = f"{root_dir}/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"

model_urls = {
    "sd_v1_5": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
    "dpt_hybrid-midas-501f0c75": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt"
}
config_path = ''

# https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
control_model_urls = {
    "control_sd15_canny": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "control_sd15_depth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "control_sd15_softedge": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",  # replaces hed, v11 uses sofftedge  model here
    "control_sd15_mlsd": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth",
    "control_sd15_normalbae": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
    "control_sd15_openpose": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "control_sd15_scribble": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
    "control_sd15_seg": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth",
    "control_sd15_temporalnet": "https://huggingface.co/CiaraRowles/TemporalNet/resolve/main/diff_control_sd15_temporalnet_fp16.safetensors",
    "control_sd15_face": "https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/control_v2p_sd15_mediapipe_face.safetensors",
    "control_sd15_ip2p": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth",
    "control_sd15_inpaint": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth",
    "control_sd15_lineart": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "control_sd15_lineart_anime": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth",
    "control_sd15_shuffle": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth"
}
# import wget
model_version = 'control_multi'  # @param ['v1','v1_inpainting','v1_instructpix2pix','v2_512','v2_depth', 'v2_768_v', "control_sd15_canny", "control_sd15_depth","control_sd15_softedge",  "control_sd15_mlsd", "control_sd15_normalbae", "control_sd15_openpose", "control_sd15_scribble", "control_sd15_seg", 'control_multi' ]
if model_version == 'v1':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inference.yaml"
if model_version == 'v1_inpainting':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1-inpainting-inference.yaml"
if model_version == 'v2_512':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference.yaml"
if model_version == 'v2_768_v':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
if model_version == 'v2_depth':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v2-midas-inference.yaml"
    os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
    if not os.path.exists(f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt"):
        midas_url = model_urls['dpt_hybrid-midas-501f0c75']
        os.makedirs(f'{root_dir}/midas_models', exist_ok=True)
        wget.download(midas_url, f"{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt")
        # !wget -O  "{root_dir}/midas_models/dpt_hybrid-midas-501f0c75.pt" https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt
control_helpers = {
    "control_sd15_canny": None,
    "control_sd15_depth": "dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_softedge": "network-bsds500.pth",
    "control_sd15_mlsd": "mlsd_large_512_fp32.pth",
    "control_sd15_normalbae": "dpt_hybrid-midas-501f0c75.pt",
    "control_sd15_openpose": ["body_pose_model.pth", "hand_pose_model.pth"],
    "control_sd15_scribble": None,
    "control_sd15_seg": "upernet_global_small.pth",
    "control_sd15_temporalnet": None,
    "control_sd15_face": None
}
if model_version == 'v1_instructpix2pix':
    config_path = f"{root_dir}/stablediffusion/configs/stable-diffusion/v1_instruct_pix2pix.yaml"
vae_ckpt = ''  # @param {'type':'string'}
if vae_ckpt == '': vae_ckpt = None
load_to = 'gpu'  # @param ['cpu','gpu']
if load_to == 'gpu': load_to = 'cuda'
quantize = True  # @param {'type':'boolean'}
no_half_vae = False  # @param {'type':'boolean'}

import gc

def load_model_from_config(config, ckpt, vae_ckpt=None, controlnet=None, verbose=False):
    with torch.no_grad():
        model = instantiate_from_config(config.model).eval().cuda()
        if no_half_vae:
            model.model.half()
            model.cond_stage_model.half()
            model.control_model.half()
        else:
            model.half()
        gc.collect()

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
        del pl_sd
        gc.collect()

        if vae_ckpt is not None:
            print(f"Loading VAE from {vae_ckpt}")
            if vae_ckpt.endswith('.safetensors'):
                vae_sd = {}
                with safe_open(vae_ckpt, framework="pt", device=load_to) as f:
                    for key in f.keys():
                        vae_sd[key] = f.get_tensor(key)
            else:
                vae_sd = torch.load(vae_ckpt, map_location=load_to)
            if "state_dict" in vae_sd:
                vae_sd = vae_sd["state_dict"]
            sd = {
                k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
                for k, v in sd.items()
            }

        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m, len(m))
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u, len(u))

        if controlnet is not None:
            ckpt = controlnet
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
            del pl_sd
            gc.collect()
            m, u = model.control_model.load_state_dict(sd, strict=False)
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m, len(m))
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u, len(u))

        return model
