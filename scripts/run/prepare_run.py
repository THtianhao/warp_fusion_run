import torch
from modules import shared
from scripts.settings.main_config import MainConfig

def prepare_run(main_config: MainConfig):
    image_prompts = {}
    controlnet_multimodel_temp = {}
    for key in main_config.controlnet_multimodel.keys():

        weight = main_config.controlnet_multimodel[key]["weight"]
        if weight != 0:
            controlnet_multimodel_temp[key] = main_config.controlnet_multimodel[key]
    main_config.controlnet_multimodel = controlnet_multimodel_temp

    main_config.inverse_mask_order = False
    can_use_sdp = hasattr(torch.nn.functional, "scaled_dot_product_attention") and callable(getattr(torch.nn.functional, "scaled_dot_product_attention"))  # not everyone has torch 2.x to use sdp
    if can_use_sdp:
        shared.opts.xformers = False
        shared.cmd_opts.xformers = False
