# @title 1.6 init main sd run function, cond_fn, color matching for SD
import os

from lpips import lpips
import torchvision.transforms as T

from scripts.model_process.model_env import device
from scripts.utils.env import root_dir, root_path

init_latent = None
target_embed = None
try:
    import Image
except:
    from PIL import Image

mask_result = False
early_stop = 0
inpainting_stop = 0
warp_interp = Image.BILINEAR

import sys
import cv2

os.chdir(root_dir)
sys.path.append(f'{root_dir}/python-color-transfer')
os.chdir(f"{root_dir}/python-color-transfer")
print(sys.path)

from python_color_transfer.color_transfer import ColorTransfer, Regrain

os.chdir(root_path)
from tqdm.auto import trange
from kornia import augmentation as KA

aug = KA.RandomAffine(0, (1 / 14, 1 / 14), p=1, padding_mode='border')

PT = ColorTransfer()
RG = Regrain()


def match_color_var(stylized_img, raw_img, opacity=1., f=PT.pdf_transfer, regrain=False):
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC)

    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    if regrain: img_arr_col = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_col = img_arr_col * opacity + img_arr_in * (1 - opacity)
    img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'), cv2.COLOR_BGR2RGB)

    return img_arr_reg


# https://gist.githubusercontent.com/trygvebw/c71334dd127d537a15e9d59790f7f5e1/raw/ed0bed6abaf75c0f1b270cf6996de3e07cbafc81/find_noise.py

import numpy as np
import k_diffusion as K

from PIL import Image
from torch import autocast
from einops import rearrange, repeat


def pil_img_to_torch(pil_img, half=False):
    image = np.array(pil_img).astype(np.float32) / 255.0
    image = rearrange(torch.from_numpy(image), 'h w c -> c h w')
    if half:
        image = image
    return (2.0 * image - 1.0).unsqueeze(0)


def pil_img_to_latent(model, img, batch_size=1, device='cuda', half=True):
    init_image = pil_img_to_torch(img, half=half).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
    if half:
        return model.get_first_stage_encoding(model.encode_first_stage(init_image))
    return model.get_first_stage_encoding(model.encode_first_stage(init_image))


import torch
from ldm.modules.midas.api import load_midas_transform

midas_tfm = load_midas_transform("dpt_hybrid")


def midas_tfm_fn(x):
    x = x = ((x + 1.0) * .5).detach().cpu().numpy()
    return midas_tfm({"image": x})["image"]


def pil2midas(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    image = midas_tfm_fn(image)
    return torch.from_numpy(image[None, ...]).float()


def find_noise_for_image(model, x, prompt, steps, cond_scale=0.0, verbose=False, normalize=True):
    with torch.no_grad():
        with autocast('cuda'):
            uncond = model.get_learned_conditioning([''])
            cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    with torch.no_grad():
        with autocast('cuda'):
            for i in trange(1, len(sigmas)):
                x_in = torch.cat([x] * 2)
                sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
                cond_in = torch.cat([uncond, cond])

                c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

                if i == 1:
                    t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
                else:
                    t = dnw.sigma_to_t(sigma_in)

                eps = model.apply_model(x_in * c_in, t, cond=cond_in)
                denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

                denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
            print(x.shape)
            if normalize:
                return (x / x.std()) * sigmas[-1]
            else:
                return x


# Based on changes suggested by briansemrau in https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/736
import hashlib

# karras noise
# https://github.com/Birch-san/stable-diffusion/blob/693c8a336aa3453d30ce403f48eb545689a679e5/scripts/txt2img_fork.py#L62-L81
sys.path.append('./k-diffusion')


def get_premature_sigma_min(steps: int, sigma_max: float, sigma_min_nominal: float, rho: float) -> float:
    min_inv_rho = sigma_min_nominal**(1 / rho)
    max_inv_rho = sigma_max**(1 / rho)
    ramp = (steps - 2) * 1 / (steps - 1)
    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho))**rho
    return sigma_min


import contextlib

none_context = contextlib.nullcontext()

pred_noise = None

diffusion_model = "stable_diffusion"

diffusion_sampling_mode = 'ddim'

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)
