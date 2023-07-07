import gc
from glob import glob

from modules import shared

def prepare_run():
    controlnet_multimodel = get_value('controlnet_multimodel', guis)
    image_prompts = {}
    controlnet_multimodel_temp = {}
    for key in controlnet_multimodel.keys():

        weight = controlnet_multimodel[key]["weight"]
        if weight != 0:
            controlnet_multimodel_temp[key] = controlnet_multimodel[key]
    controlnet_multimodel = controlnet_multimodel_temp

    inverse_mask_order = False
    can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(getattr(torch.nn.functional, "scaled_dot_product_attention"))  # not everyone has torch 2.x to use sdp
    if can_use_sdp:
        shared.opts.xformers = False
        shared.cmd_opts.xformers = False
