import os
import pickle
import sys
from torch.nn import functional as F
from guided_diffusion.nn import timestep_embedding
from resize_right import resize
from PIL import Image
from scripts.utils.env import root_dir, model_path
from scripts.video_process.color_transfor_func import writeFlow
import numpy as np

try:
    os.chdir(f'{root_dir}/src/taming-transformers')
    import taming

    os.chdir(f'{root_dir}')
    os.chdir(f'{root_dir}/k-diffusion')
    import k_diffusion

    os.chdir(f'{root_dir}')
except:
    import taming
    import k_diffusion

sys.path.append('./k-diffusion')

def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

from einops import rearrange, repeat

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            # with torch.no_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs);  # print(denoised.requires_grad)
            # with torch.enable_grad():
            # denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach();  # print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * k_diffusion.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised

    return model_fn

def make_cond_model_fn(model, cond_fn):
    def model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            # with torch.no_grad():
            # x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)  # print(denoised.requires_grad)
            # with torch.enable_grad():
            # print(sigma**0.5, sigma, sigma**2)
            denoised = denoised.detach().requires_grad_()
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()  # print(cond_grad.requires_grad)
            cond_denoised = denoised.detach() + cond_grad * k_diffusion.utils.append_dims(sigma ** 2, x.ndim)
        return cond_denoised

    return model_fn

def make_static_thresh_model_fn(model, value=1.):
    def model_fn(x, sigma, **kwargs):
        return model(x, sigma, **kwargs).clamp(-value, value)

    return model_fn

def get_image_embed(x, clip_size, clip_model):
    if x.shape[2:4] != clip_size:
        x = resize(x, out_shape=clip_size, pad_mode='reflect')
    # print('clip', x.shape)
    # x = clip_normalize(x).cuda()
    x = clip_model.encode_image(x).float()
    return F.normalize(x)

def load_img_sd(path, size):
    # print(type(path))
    # print('load_sd',path)

    image = Image.open(path).convert("RGB")
    # print(f'loaded img with size {image.size}')
    image = image.resize(size, resample=Image.LANCZOS)
    # w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

    # image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.

# @markdown If you're having crashes (CPU out of memory errors) while running this cell on standard colab env, consider saving the model as pickle.\
# @markdown You can save the pickled model on your google drive and use it instead of the usual stable diffusion model.\
# @markdown To do that, run the notebook with a high-ram env, run all cells before and including this cell as well, and save pickle in the next cell. Then you can switch to a low-ram env and load the pickled model.

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1, inpainting_mask_weight=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    if mask is not None:
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
    else:
        mask = image.new_ones(1, 1, *image.shape[-2:])

    # masked_image = image * (mask < 0.5)

    masked_image = torch.lerp(
        image,
        image * (mask < 0.5),
        inpainting_mask_weight
    )

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

import torch

# divisible by 8 fix from AUTOMATIC1111
def cat8(tensors, *args, **kwargs):
    if len(tensors) == 2:
        a, b = tensors
        if a.shape[-2:] != b.shape[-2:]:
            a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

        tensors = (a, b)

    return torch.cat(tensors, *args, **kwargs)



def save_loaded_mode(save_model_pickle, save_folder, sd_model):
    # @title Save loaded model
    # @markdown For this cell to work you need to load model in the previous cell.\
    # @markdown Saves an already loaded model as an object file, that weights less, loads faster, and requires less CPU RAM.\
    # @markdown After saving model as pickle, you can then load it as your usual stable diffusion model in thecell above.\
    # @markdown The model will be saved under the same name with .pkl extenstion.

    # save_model_pickle = False  # @param {'type':'boolean'}
    # save_folder = "/content/drive/MyDrive/models"  # @param {'type':'string'}
    if save_folder != '' and save_model_pickle:
        os.makedirs(save_folder, exist_ok=True)
        out_path = save_folder + model_path.replace('\\', '/').split('/')[-1].split('.')[0] + '.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(sd_model, f)
        print('Model successfully saved as: ', out_path)
