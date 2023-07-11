import numpy as np
import torch
import torch.nn as nn

from modules import prompt_parser
from scripts.model_process.model_env import model_version

class CFGDenoiser(nn.Module):
    def __init__(self, model, img_zero_uncond, controlnet_multimodel_mode ):
        super().__init__()
        self.inner_model = model
        self.img_zero_uncond = img_zero_uncond
        self.controlnet_multimodel_mode = controlnet_multimodel_mode

    def forward(self, x, sigma, uncond, cond, cond_scale, loaded_controlnets, image_cond=None):
        cond = prompt_parser.reconstruct_cond_batch(cond, 0)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, 0)
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        # print('cond.shape, uncond.shape', cond.shape, uncond.shape)
        cond_in = torch.cat([uncond, cond])

        if image_cond is None:
            uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
            return uncond + (cond - uncond) * cond_scale
        else:
            if model_version != 'control_multi':
                if self.img_zero_uncond:
                    img_in = torch.cat([torch.zeros_like(image_cond),
                                        image_cond])
                else:
                    img_in = torch.cat([image_cond] * 2)
                uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in],
                                                                      'c_concat': [img_in]}).chunk(2)
                return uncond + (cond - uncond) * cond_scale

            if model_version == 'control_multi' and self.controlnet_multimodel_mode != 'external':
                img_in = {}
                for key in image_cond.keys():
                    if self.img_zero_uncond or key == 'control_sd15_shuffle':
                        img_in[key] = torch.cat([torch.zeros_like(image_cond[key]),
                                                 image_cond[key]])
                    else:
                        img_in[key] = torch.cat([image_cond[key]] * 2)

                uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in],
                                                                      'c_concat': img_in,
                                                                      'controlnet_multimodel': self.controlnet_multimodel,
                                                                      'loaded_controlnets': loaded_controlnets}).chunk(2)
                return uncond + (cond - uncond) * cond_scale
            if model_version == 'control_multi' and self.controlnet_multimodel_mode == 'external':

                # wormalize weights
                weights = np.array([self.controlnet_multimodel[m]["weight"] for m in self.controlnet_multimodel.keys()])
                weights = weights / weights.sum()
                result = None
                # print(weights)
                for i, controlnet in enumerate(self.controlnet_multimodel.keys()):
                    try:
                        if self.img_zero_uncond or controlnet == 'control_sd15_shuffle':
                            img_in = torch.cat([torch.zeros_like(image_cond[controlnet]),
                                                image_cond[controlnet]])
                        else:
                            img_in = torch.cat([image_cond[controlnet]] * 2)
                    except:
                        pass
                    if weights[i] != 0:
                        controlnet_settings = self.controlnet_multimodel[controlnet]
                        self.inner_model.inner_model.control_model = loaded_controlnets[controlnet]
                        uncond, cond = self.inner_model(x_in, sigma_in, cond={"c_crossattn": [cond_in],
                                                                              'c_concat': [img_in]}).chunk(2)
                        if result is None:
                            result = (uncond + (cond - uncond) * cond_scale) * weights[i]
                        else:
                            result = result + (uncond + (cond - uncond) * cond_scale) * weights[i]
                return result
