# @title 1.6 init main sd run function, cond_fn, color matching for SD
init_latent = None
target_embed = None

import hashlib
import os

# import datetime

# (c) Alex Spirin 2023
# We use input file hashes to automate video extraction
#
def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)

    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()

    return file_hash

def get_frame_from_path_start_end_nth(video_path: str, num_frame: int, start: int = 0, end: int = 0, nth: int = 1) -> Image:
    assert os.path.exists(video_path), f"Video path or frame folder not found at {video_path}. Please specify the correct path."
    num_frame = max(0, num_frame)
    start = max(0, start)
    nth = max(1, nth)
    if os.path.isdir(video_path):
        frame_list = []
        image_extensions = ['jpg', 'png', 'tiff', 'jpeg', 'JPEG', 'bmp']
        for image_extension in image_extensions:
            flist = glob.glob(os.path.join(video_path, f'*.{image_extension}'))
            if len(flist) > 0:
                frame_list = flist
                break
        assert len(frame_list) != 0, f'No frames with {", ".join(image_extensions)} extensions found in folder {video_path}. Please specify the correct path.'
        if end == 0: end = len(frame_list)
        frame_list = frame_list[start:end:nth]
        num_frame = min(num_frame, len(frame_list))
        return PIL.Image.open(frame_list[num_frame])

    elif os.path.isfile(video_path):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            video.release()
            raise Exception(f"Error opening video file {video_path}. Please specify the correct path.")
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if end == 0: end = total_frames
        num_frame = min(num_frame, total_frames)
        frame_range = list(range(start, end, nth))
        frame_number = frame_range[num_frame]
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        if not ret:
            video.release()
            raise Exception(f"Error reading frame {frame_number} from file {video_path}.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        video.release()
        return image

import PIL

try:
    import Image
except:
    from PIL import Image

mask_result = False
early_stop = 0
inpainting_stop = 0
warp_interp = Image.BILINEAR

# init SD
from glob import glob
import argparse, os, sys
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

os.chdir(f"{root_dir}/stablediffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

os.chdir(f"{root_dir}")

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

from kornia import augmentation as KA

aug = KA.RandomAffine(0, (1 / 14, 1 / 14), p=1, padding_mode='border')
from torch.nn import functional as F

from torch.cuda.amp import GradScaler

def sd_cond_fn(x, t, denoised, init_image_sd, init_latent, init_scale,
               init_latent_scale, target_embed, consistency_mask, guidance_start_code=None,
               deflicker_fn=None, deflicker_lat_fn=None, deflicker_src=None,
               **kwargs):
    if use_scale: scaler = GradScaler()
    with torch.cuda.amp.autocast():
        global add_noise_to_latent

        # init_latent_scale,  init_scale, clip_guidance_scale, target_embed, init_latent, clamp_grad, clamp_max,
        # **kwargs):
        # global init_latent_scale
        # global init_scale
        global clip_guidance_scale
        # global target_embed
        # print(target_embed.shape)
        global clamp_grad
        global clamp_max
        loss = 0.
        if grad_denoised:
            x = denoised
            # denoised = x

            # print('grad denoised')
        grad = torch.zeros_like(x)

        processed1 = deflicker_src['processed1']
        if add_noise_to_latent:
            if t != 0:
                if guidance_use_start_code and guidance_start_code is not None:
                    noise = guidance_start_code
                else:
                    noise = torch.randn_like(x)
                noise = noise * t
                if noise_upscale_ratio > 1:
                    noise = noise[::noise_upscale_ratio, ::noise_upscale_ratio, :]
                    noise = torch.nn.functional.interpolate(noise, x.shape[2:],
                                                            mode='bilinear')
                init_latent = init_latent + noise
                if deflicker_lat_fn:
                    processed1 = deflicker_src['processed1'] + noise

        if sat_scale > 0 or init_scale > 0 or clip_guidance_scale > 0 or deflicker_scale > 0:
            with torch.autocast('cuda'):
                denoised_small = denoised[:, :, ::2, ::2]
                denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised_small)

        if clip_guidance_scale > 0:
            # compare text clip embeds with denoised image embeds
            # denoised_img = model_wrap_cfg.inner_model.inner_model.differentiable_decode_first_stage(denoised);# print(denoised.requires_grad)
            # print('d b',denoised.std(), denoised.mean())
            denoised_img = denoised_img[0].add(1).div(2)
            denoised_img = normalize(denoised_img)
            denoised_t = denoised_img.cuda()[None, ...]
            # print('d a',denoised_t.std(), denoised_t.mean())
            image_embed = get_image_embed(denoised_t)

            # image_embed = get_image_embed(denoised.add(1).div(2))
            loss = spherical_dist_loss(image_embed, target_embed).sum() * clip_guidance_scale

        if masked_guidance:
            if consistency_mask is None:
                consistency_mask = torch.ones_like(denoised)
            # consistency_mask = consistency_mask.permute(2,0,1)[None,...]
            # print(consistency_mask.shape, denoised.shape)

            consistency_mask = torch.nn.functional.interpolate(consistency_mask, denoised.shape[2:],
                                                               mode='bilinear')
            if g_invert_mask: consistency_mask = 1 - consistency_mask

        if init_latent_scale > 0:
            # compare init image latent with denoised latent
            # print(denoised.shape, init_latent.shape)

            loss += init_latent_fn(denoised, init_latent).sum() * init_latent_scale

        if sat_scale > 0:
            loss += torch.abs(denoised_img - denoised_img.clamp(min=-1, max=1)).mean()

        if init_scale > 0:
            # compare init image with denoised latent image via lpips
            # print('init_image_sd', init_image_sd)

            loss += lpips_model(denoised_img, init_image_sd[:, :, ::2, ::2]).sum() * init_scale

        if deflicker_scale > 0 and deflicker_fn is not None:
            # print('deflicker_fn(denoised_img).sum() * deflicker_scale',deflicker_fn(denoised_img).sum() * deflicker_scale)
            loss += deflicker_fn(processed2=denoised_img).sum() * deflicker_scale
            print('deflicker ', loss)

        if deflicker_latent_scale > 0 and deflicker_lat_fn is not None:
            loss += deflicker_lat_fn(processed2=denoised, processed1=processed1).sum() * deflicker_latent_scale
            print('deflicker lat', loss)

    # print('loss', loss)
    if loss != 0.:
        if use_scale:
            scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                     inputs=x)
            inv_scale = 1. / scaler.get_scale()
            grad_params = [p * inv_scale for p in scaled_grad_params]
            grad = -grad_params[0]
            # scaler.update()
        else:
            grad = -torch.autograd.grad(loss, x)[0]
        if masked_guidance:
            grad = grad * consistency_mask
        if torch.isnan(grad).any():
            print('got NaN grad')
            return torch.zeros_like(x)
        if VERBOSE: printf('loss, grad', loss, grad.max(), grad.mean(), grad.std(), denoised.mean(), denoised.std())
        if clamp_grad:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude

    return grad

import cv2

%cd
"{root_dir}/python-color-transfer"
from python_color_transfer.color_transfer import ColorTransfer, Regrain
%cd
"{root_path}/"

PT = ColorTransfer()

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

import torch
import numpy as np
# import k_diffusion as K

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

def make_depth_cond(pil_image, x):
    global frame_num
    pil_image = Image.open(pil_image).convert('RGB')
    c_cat = list()
    cc = pil2midas(pil_image).cuda()
    cc = sd_model.depth_model(cc)
    depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                   keepdim=True)
    display_depth = (cc - depth_min) / (depth_max - depth_min)
    depth_image = Image.fromarray(
        (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
    display_depth = (cc - depth_min) / (depth_max - depth_min)
    depth_image = Image.fromarray(
        (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
    if cc.shape[2:] != x.shape[2:]:
        cc = torch.nn.functional.interpolate(
            cc,
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        )
    depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                   keepdim=True)

    cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
    c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)
    # cond
    # cond = {"c_concat": [c_cat], "c_crossattn": [c]}

    # # uncond cond
    # uc_full = {"c_concat": [c_cat], "c_crossattn": [uc]}
    return c_cat, depth_image

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

def find_noise_for_image_sigma_adjustment(init_latent, prompt, image_conditioning, cfg_scale, steps, frame_num):
    rec_noise_setting_list = {
        'init_image': init_image,
        'seed': seed,
        'width': width_height[0],
        'height': width_height[1],
        'diffusion_model': diffusion_model,
        'diffusion_steps': diffusion_steps,
        'video_init_path': video_init_path,
        'extract_nth_frame': extract_nth_frame,
        'flow_video_init_path': flow_video_init_path,
        'flow_extract_nth_frame': flow_extract_nth_frame,
        'video_init_seed_continuity': video_init_seed_continuity,
        'turbo_mode': turbo_mode,
        'turbo_steps': turbo_steps,
        'turbo_preroll': turbo_preroll,
        'flow_warp': flow_warp,
        'check_consistency': check_consistency,
        'turbo_frame_skips_steps': turbo_frame_skips_steps,
        'forward_weights_clip': forward_weights_clip,
        'forward_weights_clip_turbo_step': forward_weights_clip_turbo_step,
        'padding_ratio': padding_ratio,
        'padding_mode': padding_mode,
        'consistency_blur': consistency_blur,
        'inpaint_blend': inpaint_blend,
        'match_color_strength': match_color_strength,
        'high_brightness_threshold': high_brightness_threshold,
        'high_brightness_adjust_ratio': high_brightness_adjust_ratio,
        'low_brightness_threshold': low_brightness_threshold,
        'low_brightness_adjust_ratio': low_brightness_adjust_ratio,
        'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
        'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
        'max_brightness_threshold': max_brightness_threshold,
        'min_brightness_threshold': min_brightness_threshold,
        'enable_adjust_brightness': enable_adjust_brightness,
        'dynamic_thresh': dynamic_thresh,
        'warp_interp': warp_interp,
        'reverse_cc_order': reverse_cc_order,
        'flow_lq': flow_lq,
        'use_predicted_noise': use_predicted_noise,
        'clip_guidance_scale': clip_guidance_scale,
        'clip_type': clip_type,
        'clip_pretrain': clip_pretrain,
        'missed_consistency_weight': missed_consistency_weight,
        'overshoot_consistency_weight': overshoot_consistency_weight,
        'edges_consistency_weight': edges_consistency_weight,
        'flow_blend_schedule': flow_blend_schedule,
        'steps_schedule': steps_schedule,
        'latent_scale_schedule': latent_scale_schedule,
        'flow_blend_template': flow_blend_template,
        'make_schedules': make_schedules,
        'normalize_latent': normalize_latent,
        'normalize_latent_offset': normalize_latent_offset,
        'colormatch_frame': colormatch_frame,
        'use_karras_noise': use_karras_noise,
        'end_karras_ramp_early': end_karras_ramp_early,
        'use_background_mask': use_background_mask,
        'apply_mask_after_warp': apply_mask_after_warp,
        'background': background,
        'background_source': background_source,
        'mask_source': mask_source,
        'extract_background_mask': extract_background_mask,
        'mask_video_path': mask_video_path,
        'invert_mask': invert_mask,
        'warp_strength': warp_strength,
        'flow_override_map': flow_override_map,
        'respect_sched': respect_sched,
        'color_match_frame_str': color_match_frame_str,
        'colormatch_offset': colormatch_offset,
        'latent_fixed_mean': latent_fixed_mean,
        'latent_fixed_std': latent_fixed_std,
        'colormatch_method': colormatch_method,
        'colormatch_regrain': colormatch_regrain,
        'warp_mode': warp_mode,
        'use_patchmatch_inpaiting': use_patchmatch_inpaiting,
        'blend_latent_to_init': blend_latent_to_init,
        'warp_towards_init': warp_towards_init,
        'init_grad': init_grad,
        'grad_denoised': grad_denoised,
        'colormatch_after': colormatch_after,
        'colormatch_turbo': colormatch_turbo,
        'model_version': model_version,
        'cond_image_src': cond_image_src,
        'warp_num_k': warp_num_k,
        'warp_forward': warp_forward,
        'sampler': sampler.__name__,
        'mask_clip': (mask_clip_low, mask_clip_high),
        'inpainting_mask_weight': inpainting_mask_weight,
        'inverse_inpainting_mask': inverse_inpainting_mask,
        'mask_source': mask_source,
        'model_path': model_path,
        'diff_override': diff_override,
        'image_scale_schedule': image_scale_schedule,
        'image_scale_template': image_scale_template,
        'detect_resolution': detect_resolution,
        'bg_threshold': bg_threshold,
        'diffuse_inpaint_mask_blur': diffuse_inpaint_mask_blur,
        'diffuse_inpaint_mask_thresh': diffuse_inpaint_mask_thresh,
        'add_noise_to_latent': add_noise_to_latent,
        'noise_upscale_ratio': noise_upscale_ratio,
        'fixed_seed': fixed_seed,
        'init_latent_fn': init_latent_fn.__name__,
        'value_threshold': value_threshold,
        'distance_threshold': distance_threshold,
        'masked_guidance': masked_guidance,
        'cc_masked_diffusion': cc_masked_diffusion,
        'alpha_masked_diffusion': alpha_masked_diffusion,
        'inverse_mask_order': inverse_mask_order,
        'invert_alpha_masked_diffusion': invert_alpha_masked_diffusion,
        'quantize': quantize,
        'cb_noise_upscale_ratio': cb_noise_upscale_ratio,
        'cb_add_noise_to_latent': cb_add_noise_to_latent,
        'cb_use_start_code': cb_use_start_code,
        'cb_fixed_code': cb_fixed_code,
        'cb_norm_latent': cb_norm_latent,
        'guidance_use_start_code': guidance_use_start_code,
        'controlnet_preprocess': controlnet_preprocess,
        'small_controlnet_model_path': small_controlnet_model_path,
        'use_scale': use_scale,
        'g_invert_mask': g_invert_mask,
        'controlnet_multimodel': json.dumps(controlnet_multimodel),
        'img_zero_uncond': img_zero_uncond,
        'do_softcap': do_softcap,
        'softcap_thresh': softcap_thresh,
        'softcap_q': softcap_q,
        'deflicker_latent_scale': deflicker_latent_scale,
        'deflicker_scale': deflicker_scale,
        'controlnet_multimodel_mode': controlnet_multimodel_mode,
        'no_half_vae': no_half_vae,
        'temporalnet_source': temporalnet_source,
        'temporalnet_skip_1st_frame': temporalnet_skip_1st_frame,
        'rec_randomness': rec_randomness,
        'rec_source': rec_source,
        'rec_cfg': rec_cfg,
        'rec_prompts': rec_prompts,
        'inpainting_mask_source': inpainting_mask_source,
        'rec_steps_pct': rec_steps_pct,
        'max_faces': max_faces,
        'num_flow_updates': num_flow_updates,
        'control_sd15_openpose_hands_face': control_sd15_openpose_hands_face,
        'control_sd15_depth_detector': control_sd15_openpose_hands_face,
        'control_sd15_softedge_detector': control_sd15_softedge_detector,
        'control_sd15_seg_detector': control_sd15_seg_detector,
        'control_sd15_scribble_detector': control_sd15_scribble_detector,
        'control_sd15_lineart_coarse': control_sd15_lineart_coarse,
        'control_sd15_inpaint_mask_source': control_sd15_inpaint_mask_source,
        'control_sd15_shuffle_source': control_sd15_shuffle_source,
        'control_sd15_shuffle_1st_source': control_sd15_shuffle_1st_source,
        'consistency_dilate': consistency_dilate
    }
    settings_hash = hashlib.sha256(json.dumps(rec_noise_setting_list).encode('utf-8')).hexdigest()[:16]
    filepath = f'{recNoiseCacheFolder}/{settings_hash}_{frame_num:06}.pt'
    if os.path.exists(filepath) and not overwrite_rec_noise:
        print(filepath)
        noise = torch.load(filepath)
        print('loading existing noise')
        return noise
    steps = int(copy.copy(steps) * rec_steps_pct)
    cond = prompt_parser.get_learned_conditioning(sd_model, prompt, steps)
    uncond = prompt_parser.get_learned_conditioning(sd_model, [''], steps)
    cfg_scale = rec_cfg
    cond = prompt_parser.reconstruct_cond_batch(cond, 0)
    uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)

    x = init_latent

    s_in = x.new_ones([x.shape[0]])
    if sd_model.parameterization == "v":
        dnw = K.external.CompVisVDenoiser(sd_model)
        skip = 1
    else:
        dnw = K.external.CompVisDenoiser(sd_model)
        skip = 0
    sigmas = dnw.get_sigmas(steps).flip(0)

    for i in trange(1, len(sigmas)):

        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        # image_conditioning = torch.cat([image_conditioning] * 2)
        # cond_in = {"c_concat": [image_conditioning], "c_crossattn": [cond_in]}
        if model_version == 'control_multi' and controlnet_multimodel_mode == 'external':
            raise Exception("Predicted noise not supported for external mode. Please turn predicted noise off or use internal mode.")
        if image_conditioning is not None:
            if model_version != 'control_multi':
                if img_zero_uncond:
                    img_in = torch.cat([torch.zeros_like(image_conditioning),
                                        image_conditioning])
                else:
                    img_in = torch.cat([image_conditioning] * 2)
                cond_in = {"c_crossattn": [cond_in], 'c_concat': [img_in]}

            if model_version == 'control_multi' and controlnet_multimodel_mode != 'external':
                img_in = {}
                for key in image_conditioning.keys():
                    img_in[key] = torch.cat([torch.zeros_like(image_conditioning[key]),
                                             image_conditioning[key]]) if img_zero_uncond else torch.cat([image_conditioning[key]] * 2)

                cond_in = {"c_crossattn": [cond_in], 'c_concat': img_in,
                           'controlnet_multimodel': controlnet_multimodel,
                           'loaded_controlnets': loaded_controlnets}

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)[skip:]]

        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
        else:
            t = dnw.sigma_to_t(sigma_in)

        eps = sd_model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cfg_scale

        if i == 1:
            d = (x - denoised) / (2 * sigmas[i])
        else:
            d = (x - denoised) / sigmas[i - 1]

        dt = sigmas[i] - sigmas[i - 1]
        x = x + d * dt

        # This shouldn't be necessary, but solved some VRAM issues
        del x_in, sigma_in, cond_in, c_out, c_in, t,
        del eps, denoised_uncond, denoised_cond, denoised, d, dt

    # return (x / x.std()) * sigmas[-1]
    x = x / sigmas[-1]
    torch.save(x, filepath)
    return x  # / sigmas[-1]

# karras noise
# https://github.com/Birch-san/stable-diffusion/blob/693c8a336aa3453d30ce403f48eb545689a679e5/scripts/txt2img_fork.py#L62-L81
sys.path.append('./k-diffusion')

def get_premature_sigma_min(
        steps: int,
        sigma_max: float,
        sigma_min_nominal: float,
        rho: float
) -> float:
    min_inv_rho = sigma_min_nominal ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    ramp = (steps - 2) * 1 / (steps - 1)
    sigma_min = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigma_min

import contextlib

none_context = contextlib.nullcontext()

def masked_callback(args, callback_steps, masks, init_latent, start_code):
    # print('callback_step', callback_step)
    # print([o.shape for o in masks])
    init_latent = init_latent.clone()
    # print(args['i'])
    masks = [m[:, 0:1, ...] for m in masks]
    # print(args['x'].shape, mask.shape)
    final_mask = None  # create a combined mask for this step
    for (mask, callback_step) in zip(masks, callback_steps):

        if args['i'] <= callback_step:
            mask = torch.nn.functional.interpolate(mask, args['x'].shape[2:],
                                                   mode='bilinear')
            if final_mask is None:
                final_mask = mask
            else:
                final_mask = final_mask * mask

    mask = final_mask

    if mask is not None:
        # PIL.Image.fromarray(np.repeat(mask.clone().cpu().numpy()[0,0,...][...,None],3, axis=2).astype('uint8')*255).save(f'{root_dir}/{args["i"]}.jpg')
        if cb_use_start_code:
            noise = start_code
        else:
            noise = torch.randn_like(args['x'])
        noise = noise * args['sigma']
        if cb_noise_upscale_ratio > 1:
            noise = noise[::noise_upscale_ratio, ::noise_upscale_ratio, :]
            noise = torch.nn.functional.interpolate(noise, args['x'].shape[2:],
                                                    mode='bilinear')
        # mask = torch.nn.functional.interpolate(mask, args['x'].shape[2:],
        #                                                 mode='bilinear')
        if VERBOSE: print('Applying callback at step ', args['i'])
        if cb_add_noise_to_latent:
            init_latent = init_latent + noise
        if cb_norm_latent:
            noise = init_latent
            noise2 = args['x']
            n_mean = noise2.mean(dim=(2, 3), keepdim=True)
            n_std = noise2.std(dim=(2, 3), keepdim=True)
            n2_mean = noise.mean(dim=(2, 3), keepdim=True)
            noise = noise - (n2_mean - n_mean)
            n2_std = noise.std(dim=(2, 3), keepdim=True)
            noise = noise / (n2_std / n_std)
            init_latent = noise

        args['x'] = args['x'] * (1 - mask) + (init_latent) * mask  # ok
    # args['x'] = args['x']*(mask) + (init_latent)*(1-mask) #test reverse
    # return args['x']

    return args['x']

pred_noise = None

def run_sd(opt, init_image, skip_timesteps, H, W, text_prompt, neg_prompt, steps, seed,
           init_scale, init_latent_scale, cond_image, cfg_scale, image_scale,
           cond_fn=None, init_grad_img=None, consistency_mask=None, frame_num=0,
           deflicker_src=None, prev_frame=None, rec_prompt=None, rec_frame=None,
           control_inpainting_mask=None, shuffle_source=None, ref_image=None, alpha_mask=None):
    # sampler = sample_euler
    seed_everything(seed)
    sd_model.cuda()
    # global cfg_scale
    if VERBOSE:
        print('seed', 'clip_guidance_scale', 'init_scale', 'init_latent_scale', 'clamp_grad', 'clamp_max',
              'init_image', 'skip_timesteps', 'cfg_scale')
        print(seed, clip_guidance_scale, init_scale, init_latent_scale, clamp_grad,
              clamp_max, init_image, skip_timesteps, cfg_scale)
    global start_code, inpainting_mask_weight, inverse_inpainting_mask, start_code_cb, guidance_start_code
    global pred_noise, controlnet_preprocess
    # global frame_num
    global normalize_latent
    global first_latent
    global first_latent_source
    global use_karras_noise
    global end_karras_ramp_early
    global latent_fixed_norm
    global latent_norm_4d
    global latent_fixed_mean
    global latent_fixed_std
    global n_mean_avg
    global n_std_avg
    global reference_latent

    batch_size = num_samples = 1
    scale = cfg_scale

    C = 4  # 4
    f = 8  # 8
    H = H
    W = W
    if VERBOSE: print(W, H, 'WH')
    prompt = text_prompt[0]

    neg_prompt = neg_prompt[0]
    ddim_steps = steps

    # init_latent_scale = 0. #20
    prompt_clip = prompt

    assert prompt is not None
    prompts = [prompt]

    if VERBOSE: print('prompts', prompts, text_prompt)

    precision_scope = autocast

    t_enc = ddim_steps - skip_timesteps

    if init_image is not None:
        if isinstance(init_image, str):
            if not init_image.endswith('_lat.pt'):
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        init_image_sd = load_img_sd(init_image, size=(W, H)).cuda()
                        init_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(init_image_sd))
                        x0 = init_latent
            if init_image.endswith('_lat.pt'):
                init_latent = torch.load(init_image).cuda()
                init_image_sd = None
                x0 = init_latent

    reference_latent = None
    if ref_image is not None and reference_active:
        if os.path.exists(ref_image):
            with torch.no_grad(), torch.cuda.amp.autocast():
                reference_img = load_img_sd(ref_image, size=(W, H)).cuda()
                reference_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(reference_img))
        else:
            print('Failed to load reference image')
            ref_image = None

    if use_predicted_noise:
        if rec_frame is not None:
            with torch.cuda.amp.autocast():
                rec_frame_img = load_img_sd(rec_frame, size=(W, H)).cuda()
                rec_frame_latent = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(rec_frame_img))

    if init_grad_img is not None:
        print('Replacing init image for cond fn')
        init_image_sd = load_img_sd(init_grad_img, size=(W, H)).cuda()

    if blend_latent_to_init > 0. and first_latent is not None:
        print('Blending to latent ', first_latent_source)
        x0 = x0 * (1 - blend_latent_to_init) + blend_latent_to_init * first_latent
    if normalize_latent != 'off' and first_latent is not None:
        if VERBOSE:
            print('norm to 1st latent')
            print('latent source - ', first_latent_source)
        # noise2 - target
        # noise - modified

        if latent_norm_4d:
            n_mean = first_latent.mean(dim=(2, 3), keepdim=True)
            n_std = first_latent.std(dim=(2, 3), keepdim=True)
        else:
            n_mean = first_latent.mean()
            n_std = first_latent.std()

        if n_mean_avg is None and n_std_avg is None:
            n_mean_avg = n_mean.clone().detach().cpu().numpy()[0, :, 0, 0]
            n_std_avg = n_std.clone().detach().cpu().numpy()[0, :, 0, 0]
        else:
            n_mean_avg = n_mean_avg * n_smooth + (1 - n_smooth) * n_mean.clone().detach().cpu().numpy()[0, :, 0, 0]
            n_std_avg = n_std_avg * n_smooth + (1 - n_smooth) * n_std.clone().detach().cpu().numpy()[0, :, 0, 0]

        if VERBOSE:
            print('n_stats_avg (mean, std): ', n_mean_avg, n_std_avg)
        if normalize_latent == 'user_defined':
            n_mean = latent_fixed_mean
            if isinstance(n_mean, list) and len(n_mean) == 4: n_mean = np.array(n_mean)[None, :, None, None]
            n_std = latent_fixed_std
            if isinstance(n_std, list) and len(n_std) == 4: n_std = np.array(n_std)[None, :, None, None]
        if latent_norm_4d:
            n2_mean = x0.mean(dim=(2, 3), keepdim=True)
        else:
            n2_mean = x0.mean()
        x0 = x0 - (n2_mean - n_mean)
        if latent_norm_4d:
            n2_std = x0.std(dim=(2, 3), keepdim=True)
        else:
            n2_std = x0.std()
        x0 = x0 / (n2_std / n_std)

    if clip_guidance_scale > 0:
        # text_features = clip_model.encode_text(text)
        target_embed = F.normalize(clip_model.encode_text(open_clip.tokenize(prompt_clip).cuda()).float())
    else:
        target_embed = None

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with precision_scope("cuda"):
                scope = none_context if model_version == 'v1_inpainting' else sd_model.ema_scope()
                with scope:
                    tic = time.time()
                    all_samples = []
                    uc = None
                    if True:
                        if scale != 1.0:
                            uc = prompt_parser.get_learned_conditioning(sd_model, [neg_prompt], ddim_steps)

                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = prompt_parser.get_learned_conditioning(sd_model, prompts, ddim_steps)

                        shape = [C, H // f, W // f]
                        if use_karras_noise:

                            rho = 7.
                            # 14.6146
                            sigma_max = model_wrap.sigmas[-1].item()
                            sigma_min_nominal = model_wrap.sigmas[0].item()
                            # get the "sigma before sigma_min" from a slightly longer ramp
                            # https://github.com/crowsonkb/k-diffusion/pull/23#issuecomment-1234872495
                            premature_sigma_min = get_premature_sigma_min(
                                steps=steps + 1,
                                sigma_max=sigma_max,
                                sigma_min_nominal=sigma_min_nominal,
                                rho=rho
                            )
                            sigmas = K.sampling.get_sigmas_karras(
                                n=steps,
                                sigma_min=premature_sigma_min if end_karras_ramp_early else sigma_min_nominal,
                                sigma_max=sigma_max,
                                rho=rho,
                                device='cuda',
                            )
                        else:
                            sigmas = model_wrap.get_sigmas(ddim_steps)
                        alpha_mask_t = None
                        if alpha_mask is not None and init_image is not None:
                            print('alpha_mask.shape', alpha_mask.shape)
                            alpha_mask_t = torch.from_numpy(alpha_mask).float().to(init_latent.device)[None, None, ...][:, 0:1, ...]
                        consistency_mask_t = None
                        if consistency_mask is not None and init_image is not None:
                            consistency_mask_t = torch.from_numpy(consistency_mask).float().to(init_latent.device).permute(2, 0, 1)[None, ...][:, 0:1, ...]
                        if guidance_use_start_code:
                            guidance_start_code = torch.randn_like(init_latent)

                        deflicker_fn = deflicker_lat_fn = None
                        if frame_num > args.start_frame:
                            def absdiff(a, b):
                                return abs(a - b)

                            for key in deflicker_src.keys():
                                deflicker_src[key] = load_img_sd(deflicker_src[key], size=(W, H)).cuda()
                            deflicker_fn = partial(deflicker_loss, processed1=deflicker_src['processed1'][:, :, ::2, ::2],
                                                   raw1=deflicker_src['raw1'][:, :, ::2, ::2], raw2=deflicker_src['raw2'][:, :, ::2, ::2], criterion1=absdiff, criterion2=lpips_model)
                            for key in deflicker_src.keys():
                                deflicker_src[key] = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(deflicker_src[key]))
                            deflicker_lat_fn = partial(deflicker_loss,
                                                       raw1=deflicker_src['raw1'], raw2=deflicker_src['raw2'], criterion1=absdiff, criterion2=rmse)
                        cond_fn_partial = partial(sd_cond_fn, init_image_sd=init_image_sd,
                                                  init_latent=init_latent,
                                                  init_scale=init_scale,
                                                  init_latent_scale=init_latent_scale,
                                                  target_embed=target_embed,
                                                  consistency_mask=consistency_mask_t,
                                                  start_code=guidance_start_code,
                                                  deflicker_fn=deflicker_fn, deflicker_lat_fn=deflicker_lat_fn, deflicker_src=deflicker_src
                                                  )
                        callback_partial = None
                        if cc_masked_diffusion and consistency_mask is not None or alpha_masked_diffusion and alpha_mask is not None:
                            if cb_fixed_code:
                                if start_code_cb is None:
                                    if VERBOSE: print('init start code')
                                    start_code_cb = torch.randn_like(x0)
                            else:
                                start_code_cb = torch.randn_like(x0)
                            # start_code = torch.randn_like(x0)
                            callback_steps = []
                            callback_masks = []
                            if cc_masked_diffusion and consistency_mask is not None:
                                callback_masks.append(consistency_mask_t)
                                callback_steps.append(int((ddim_steps - skip_timesteps) * cc_masked_diffusion))
                            if alpha_masked_diffusion and alpha_mask is not None:
                                if invert_alpha_masked_diffusion:
                                    alpha_mask_t = 1. - alpha_mask_t
                                callback_masks.append(alpha_mask_t)
                                callback_steps.append(int((ddim_steps - skip_timesteps) * alpha_masked_diffusion))
                            if inverse_mask_order:
                                callback_masks.reverse()
                                callback_steps.reverse()

                            if VERBOSE: print('callback steps', callback_steps)
                            callback_partial = partial(masked_callback,
                                                       callback_steps=callback_steps,
                                                       masks=callback_masks,
                                                       init_latent=init_latent, start_code=start_code_cb)
                        if new_prompt_loras == {}:
                            # only use cond fn when loras are off
                            model_fn = make_cond_model_fn(model_wrap_cfg, cond_fn_partial)
                            # model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
                        else:
                            model_fn = model_wrap_cfg

                        model_fn = make_static_thresh_model_fn(model_fn, dynamic_thresh)
                        depth_img = None
                        depth_cond = None
                        if model_version == 'v2_depth':
                            if VERBOSE: print('using depth')
                            depth_cond, depth_img = make_depth_cond(cond_image, x0)
                        if 'control_' in model_version:
                            input_image = np.array(Image.open(cond_image).resize(size=(W, H)));  # print(type(input_image), 'input_image', input_image.shape)

                        detected_maps = {}
                        if model_version == 'control_multi':
                            if offload_model:
                                for key in loaded_controlnets.keys():
                                    loaded_controlnets[key].cuda()

                            models = list(controlnet_multimodel.keys());
                            print(models)
                        else:
                            models = model_version
                        if not controlnet_preprocess and 'control_' in model_version:
                            # if multiple cond models without preprocessing - add input to all models
                            if model_version == 'control_multi':
                                for i in models:
                                    detected_map = input_image
                                    if i in ['control_sd15_normal']:
                                        detected_map = detected_map[:, :, ::-1]
                                    detected_maps[i] = detected_map
                            else:
                                detected_maps[model_version] = input_image

                        if 'control_sd15_temporalnet' in models:
                            if prev_frame is not None:
                                # prev_frame = cond_image
                                detected_map = np.array(Image.open(prev_frame).resize(size=(W, H)));  # print(type(input_image), 'input_image', input_image.shape)
                                detected_maps['control_sd15_temporalnet'] = detected_map
                            else:

                                if VERBOSE: print('skipping control_sd15_temporalnet as prev_frame is None')
                                models = [o for o in models if o != 'control_sd15_temporalnet']
                                if VERBOSE: print('models after removing temp', models)

                        if controlnet_preprocess and 'control_' in model_version:
                            if 'control_sd15_face' in models:

                                detected_map = generate_annotation(input_image, max_faces)
                                if detected_map is not None:
                                    detected_maps['control_sd15_face'] = detected_map
                                else:
                                    if VERBOSE: print('No faces detected')
                                    models = [o for o in models if o != 'control_sd15_face']

                            if 'control_sd15_normal' in models:
                                if offload_model: apply_depth.model.cuda()
                                input_image = HWC3(np.array(input_image));
                                print(type(input_image))

                                input_image = resize_image(input_image, detect_resolution);
                                print((input_image.dtype))
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    _, detected_map = apply_depth(input_image, bg_th=bg_threshold)
                                detected_map = HWC3(detected_map)
                                if offload_model: apply_depth.model.cpu()

                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                                detected_maps['control_sd15_normal'] = detected_map

                            if 'control_sd15_normalbae' in models:
                                if offload_model: apply_normal.model.cuda()
                                input_image = HWC3(np.array(input_image));
                                print(type(input_image))

                                input_image = resize_image(input_image, detect_resolution);
                                print((input_image.dtype))
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_normal(input_image)  # , bg_th=bg_threshold)
                                detected_map = HWC3(detected_map)
                                if offload_model: apply_normal.model.cpu()

                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
                                detected_maps['control_sd15_normalbae'] = detected_map

                            if 'control_sd15_depth' in models:
                                if offload_model: apply_depth.model.cuda()
                                input_image = HWC3(np.array(input_image));  # print(type(input_image))
                                Image.fromarray(input_image.astype('uint8')).save('./test.jpg')
                                input_image = resize_image(input_image, detect_resolution);  # print((input_image.dtype), input_image.shape, input_image.size)

                                if control_sd15_depth_detector == 'Midas':
                                    with torch.cuda.amp.autocast(True), torch.no_grad():
                                        detected_map, _ = apply_depth(input_image)
                                if control_sd15_depth_detector == 'Zoe':
                                    with torch.cuda.amp.autocast(False), torch.no_grad():
                                        # apply_depth.model.load_state_dict(torch.load('/content/ControlNet/annotator/ckpts/ZoeD_M12_N.pt')['model'])
                                        detected_map = apply_depth(input_image)
                                # print('dectected map depth',detected_map.shape, detected_map.min(), detected_map.max(), detected_map.mean(), detected_map.std(),  )
                                detected_map = HWC3(detected_map)
                                if offload_model: apply_depth.model.cpu()
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                detected_maps['control_sd15_depth'] = detected_map

                            if 'control_sd15_canny' in models:
                                img = HWC3(input_image)

                                # H, W, C = img.shape

                                detected_map = apply_canny(img, low_threshold, high_threshold)
                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps['control_sd15_canny'] = detected_map

                            if 'control_sd15_softedge' in models:
                                if offload_model: apply_softedge.netNetwork.cuda()
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_softedge(resize_image(input_image, detect_resolution))
                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                detected_maps['control_sd15_softedge'] = detected_map
                                if offload_model: apply_softedge.netNetwork.cpu()

                            if 'control_sd15_mlsd' in models:
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_mlsd(resize_image(input_image, detect_resolution), value_threshold, distance_threshold)
                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps['control_sd15_mlsd'] = detected_map

                            if 'control_sd15_openpose' in models:
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_openpose(resize_image(input_image,
                                                                               detect_resolution), hand_and_face=control_sd15_openpose_hands_face)

                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps['control_sd15_openpose'] = detected_map

                            if 'control_sd15_scribble' in models:
                                input_image = HWC3(input_image)
                                # H, W, C = img.shape

                                if offload_model: apply_scribble.netNetwork.cuda()
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_scribble(resize_image(input_image, detect_resolution))

                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
                                detected_map = nms(detected_map, 127, 3.0)
                                detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
                                detected_map[detected_map > 4] = 255
                                detected_map[detected_map < 255] = 0
                                detected_maps['control_sd15_scribble'] = detected_map
                                if offload_model: apply_scribble.netNetwork.cpu()

                            if "control_sd15_seg" in models:
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_seg(resize_image(input_image, detect_resolution))

                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps["control_sd15_seg"] = detected_map

                            if "control_sd15_lineart" in models:
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_lineart(resize_image(input_image, detect_resolution), coarse=control_sd15_lineart_coarse)

                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps["control_sd15_lineart"] = detected_map

                            if "control_sd15_lineart_anime" in models:
                                input_image = HWC3(input_image)
                                with torch.cuda.amp.autocast(True), torch.no_grad():
                                    detected_map = apply_lineart_anime(resize_image(input_image, detect_resolution))

                                detected_map = HWC3(detected_map)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps["control_sd15_lineart_anime"] = detected_map

                            if "control_sd15_ip2p" in models:
                                input_image = HWC3(input_image)
                                detected_map = input_image.copy()
                                img = resize_image(input_image, detect_resolution)
                                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                                detected_maps["control_sd15_ip2p"] = detected_map

                            if "control_sd15_shuffle" in models:
                                shuffle_image = np.array(Image.open(shuffle_source))
                                shuffle_image = HWC3(shuffle_image)
                                shuffle_image = cv2.resize(shuffle_image, (W, H), interpolation=cv2.INTER_NEAREST)
                                # shuffle_image = resize_image(shuffle_image, detect_resolution)

                                dH, dW, dC = shuffle_image.shape
                                detected_map = apply_shuffle(shuffle_image, w=dW, h=dH, f=256)
                                detected_maps["control_sd15_shuffle"] = detected_map

                            if "control_sd15_inpaint" in models:

                                if control_inpainting_mask is None:
                                    if VERBOSE: print('skipping control_sd15_inpaint as control_inpainting_mask is None')
                                    models = [o for o in models if o != 'control_sd15_inpaint']
                                    if VERBOSE: print('models after removing temp', models)
                                else:
                                    control_inpainting_mask *= 255
                                    control_inpainting_mask = 255 - control_inpainting_mask
                                    if VERBOSE: print('control_inpainting_mask', control_inpainting_mask.shape, control_inpainting_mask.min(), control_inpainting_mask.max())
                                    if VERBOSE: print('control_inpainting_mask', (control_inpainting_mask[..., 0] == control_inpainting_mask[..., 0]).mean())
                                    img = np.array(Image.open(init_image).resize(size=(W, H)))
                                    h, w, C = img.shape
                                    # contolnet inpaint mask - H, W, 0-255 np array
                                    detected_mask = cv2.resize(control_inpainting_mask[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
                                    detected_map = img.astype(np.float32).copy()
                                    detected_map[detected_mask > 127] = -255.0  # use -1 as inpaint value
                                    detected_maps["control_sd15_inpaint"] = detected_map

                        if 'control_' in model_version:
                            gc.collect()
                            torch.cuda.empty_cache()
                            gc.collect()
                            if VERBOSE: print('Postprocessing cond maps')

                            def postprocess_map(detected_map):
                                control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
                                control = torch.stack([control for _ in range(num_samples)], dim=0)
                                depth_cond = einops.rearrange(control, 'b h w c -> b c h w').clone()
                                # if VERBOSE: print('depth_cond', depth_cond.min(), depth_cond.max(), depth_cond.mean(), depth_cond.std(), depth_cond.shape)
                                return depth_cond

                            if model_version == 'control_multi':
                                print('init shape', init_latent.shape, H, W)
                                for m in models:
                                    if save_controlnet_annotations:
                                        PIL.Image.fromarray(detected_maps[m].astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_{m}_{frame_num:06}.jpg', quality=95)
                                    detected_maps[m] = postprocess_map(detected_maps[m])
                                    if VERBOSE: print('detected_maps[m].shape', m, detected_maps[m].shape)

                                depth_cond = detected_maps
                            else:
                                depth_cond = postprocess_map(detected_maps[model_version])

                        if model_version == 'v1_instructpix2pix':
                            if isinstance(cond_image, str):
                                print('Got img cond: ', cond_image)
                                with torch.no_grad():
                                    with torch.cuda.amp.autocast():
                                        input_image = Image.open(cond_image).resize(size=(W, H))
                                        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                                        input_image = rearrange(input_image, "h w c -> 1 c h w").to(sd_model.device)
                                        depth_cond = sd_model.encode_first_stage(input_image).mode()

                        if model_version == 'v1_inpainting':
                            print('using inpainting')
                            if cond_image is not None:
                                if inverse_inpainting_mask: cond_image = 1 - cond_image
                                cond_image = Image.fromarray((cond_image * 255).astype('uint8'))

                            batch = make_batch_sd(Image.open(init_image).resize((W, H)), cond_image, txt=prompt, device=device, num_samples=1, inpainting_mask_weight=inpainting_mask_weight)
                            c_cat = list()
                            for ck in sd_model.concat_keys:
                                cc = batch[ck].float()
                                if ck != sd_model.masked_image_key:

                                    cc = torch.nn.functional.interpolate(cc, scale_factor=1 / 8)
                                else:
                                    cc = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(cc))
                                c_cat.append(cc)
                            depth_cond = torch.cat(c_cat, dim=1)
                        # print('depth cond', depth_cond)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': scale,
                                      'image_cond': depth_cond}
                        if model_version == 'v1_instructpix2pix':
                            extra_args['image_scale'] = image_scale
                            # extra_args['cond'] = sd_model.get_learned_conditioning(prompts)
                            # extra_args['uncond'] = sd_model.get_learned_conditioning([""])
                        if skip_timesteps > 0:
                            # using non-random start code
                            if fixed_code:
                                if start_code is None:
                                    if VERBOSE: print('init start code')
                                    start_code = torch.randn_like(x0)
                                if not use_legacy_fixed_code:
                                    rand_code = torch.randn_like(x0)
                                    combined_code = ((1 - code_randomness) * start_code + code_randomness * rand_code) / ((code_randomness ** 2 + (1 - code_randomness) ** 2) ** 0.5)
                                    noise = combined_code - (x0 / sigmas[0])
                                    noise = noise * sigmas[ddim_steps - t_enc - 1]

                                # older version
                                if use_legacy_fixed_code:
                                    normalize_code = True
                                    if normalize_code:
                                        noise2 = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]
                                        if latent_norm_4d:
                                            n_mean = noise2.mean(dim=(2, 3), keepdim=True)
                                        else:
                                            n_mean = noise2.mean()
                                        if latent_norm_4d:
                                            n_std = noise2.std(dim=(2, 3), keepdim=True)
                                        else:
                                            n_std = noise2.std()

                                    noise = torch.randn_like(x0)
                                    noise = (start_code * (1 - code_randomness) + (code_randomness) * noise) * sigmas[ddim_steps - t_enc - 1]
                                    if normalize_code:
                                        if latent_norm_4d:
                                            n2_mean = noise.mean(dim=(2, 3), keepdim=True)
                                        else:
                                            n2_mean = noise.mean()
                                        noise = noise - (n2_mean - n_mean)
                                        if latent_norm_4d:
                                            n2_std = noise.std(dim=(2, 3), keepdim=True)
                                        else:
                                            n2_std = noise.std()
                                        noise = noise / (n2_std / n_std)

                            else:
                                noise = torch.randn_like(x0) * sigmas[ddim_steps - t_enc - 1]  # correct one
                                if use_predicted_noise:
                                    print('using predicted noise')
                                    rand_noise = torch.randn_like(x0)
                                    rec_noise = find_noise_for_image_sigma_adjustment(init_latent=rec_frame_latent, prompt=rec_prompt, image_conditioning=depth_cond, cfg_scale=scale, steps=ddim_steps,
                                                                                      frame_num=frame_num)
                                    combined_noise = ((1 - rec_randomness) * rec_noise + rec_randomness * rand_noise) / ((rec_randomness ** 2 + (1 - rec_randomness) ** 2) ** 0.5)
                                    noise = combined_noise - (x0 / sigmas[0])
                                    noise = noise * sigmas[ddim_steps - t_enc - 1]  # faster collapse

                                print('noise')
                                # noise = noise[::4,::4,:]
                                # noise = torch.nn.functional.interpolate(noise, scale_factor=4, mode='bilinear')
                            if t_enc != 0:
                                xi = x0 + noise
                                # printf('xi', xi.shape, xi.min().item(), xi.max().item(), xi.std().item(), xi.mean().item())
                                # print(xi.mean(), xi.std(), xi.min(), xi.max())
                                sigma_sched = sigmas[ddim_steps - t_enc - 1:]
                                # sigma_sched = sigmas[ddim_steps - t_enc:]
                                samples_ddim = sampler(model_fn, xi, sigma_sched, extra_args=extra_args, callback=callback_partial)
                            else:
                                samples_ddim = x0
                        else:
                            # if use_predicted_noise and frame_num>0:
                            if use_predicted_noise:
                                print('using predicted noise')
                                rand_noise = torch.randn_like(x0)
                                rec_noise = find_noise_for_image_sigma_adjustment(init_latent=rec_frame_latent, prompt=rec_prompt, image_conditioning=depth_cond, cfg_scale=scale, steps=ddim_steps,
                                                                                  frame_num=frame_num)
                                combined_noise = ((1 - rec_randomness) * rec_noise + rec_randomness * rand_noise) / ((rec_randomness ** 2 + (1 - rec_randomness) ** 2) ** 0.5)
                                x = combined_noise  # - (x0 / sigmas[0])
                            else:
                                x = torch.randn([batch_size, *shape], device=device)
                            x = x * sigmas[0]
                            samples_ddim = sampler(model_fn, x, sigmas, extra_args=extra_args, callback=callback_partial)
                        if first_latent is None:
                            if VERBOSE: print('setting 1st latent')
                            first_latent_source = 'samples ddim (1st frame output)'
                            first_latent = samples_ddim

                        if offload_model:
                            sd_model.model.cpu()
                            sd_model.cond_stage_model.cpu()
                            if model_version == 'control_multi':
                                for key in loaded_controlnets.keys():
                                    loaded_controlnets[key].cpu()

                        gc.collect()
                        torch.cuda.empty_cache()
                        x_samples_ddim = sd_model.decode_first_stage(samples_ddim)
                        printf('x_samples_ddim', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())
                        scale_raw_sample = False
                        if scale_raw_sample:
                            m = x_samples_ddim.mean()
                            x_samples_ddim -= m;
                            r = (x_samples_ddim.max() - x_samples_ddim.min()) / 2

                            x_samples_ddim /= r
                            x_samples_ddim += m;
                            if VERBOSE: printf('x_samples_ddim scaled', x_samples_ddim.min(), x_samples_ddim.max(), x_samples_ddim.std(), x_samples_ddim.mean())

                        all_samples.append(x_samples_ddim)
    return all_samples, samples_ddim, depth_img

diffusion_model = "stable_diffusion"

diffusion_sampling_mode = 'ddim'

normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
lpips_model = lpips.LPIPS(net='vgg').to(device)
