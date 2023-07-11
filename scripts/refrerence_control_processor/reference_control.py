# Attention Injection by Lvmin Zhang
# https://github.com/Mikubill/sd-webui-controlnet
import torch

from ldm.modules.diffusionmodules.util import timestep_embedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import BasicTransformerBlock
from enum import Enum

from scripts.refrerence_control_processor.reference_config import ReferenceConfig

class AttentionAutoMachine(Enum):
    """
    Lvmin's algorithm for Attention AutoMachine States.
    """
    Read = "Read"
    Write = "Write"

# DFS Search for Torch.nn.Module, Written by Lvmin
def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

import inspect, re

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def reference_control(config: ReferenceConfig, sd_model, reference_latent):
    outer = sd_model.model.diffusion_model


    # Attention Injection by Lvmin Zhang
    # https://github.com/lllyasviel
    # https://github.com/Mikubill/sd-webui-controlnet
    def control_forward(x, timesteps=None, context=None, control=None, only_mid_control=False, self=outer, **kwargs):
        if reference_latent is not None:
            # print('Using reference')
            query_size = int(x.shape[0])
            used_hint_cond_latent = reference_latent
            uc_mask = torch.tensor([0, 1], dtype=x.dtype, device=x.device)[:, None, None, None]
            ref_cond_xt = sd_model.q_sample(used_hint_cond_latent, torch.round(timesteps.float()).long())

            if config.reference_mode == 'Controlnet':
                ref_uncond_xt = x.clone()
                # print('ControlNet More Important -  Using standard cfg for reference.')
            elif config.reference_mode == 'Prompt':
                ref_uncond_xt = ref_cond_xt.clone()
                # print('Prompt More Important -  Using no cfg for reference.')
            else:
                ldm_time_max = getattr(sd_model, 'num_timesteps', 1000)
                time_weight = (timesteps.float() / float(ldm_time_max)).clip(0, 1)[:, None, None, None]
                time_weight *= torch.pi * 0.5
                # We should use sin/cos to make sure that the std of weighted matrix follows original ddpm schedule
                ref_uncond_xt = x * torch.sin(time_weight) + ref_cond_xt.clone() * torch.cos(time_weight)
                # print('Balanced - Using time-balanced cfg for reference.')
            for module in outer.attn_module_list:
                module.bank = []
            ref_xt = ref_cond_xt * uc_mask + ref_uncond_xt * (1 - uc_mask)
            outer.attention_auto_machine = AttentionAutoMachine.Write
            # print('ok')
            outer.original_forward(x=ref_xt, timesteps=timesteps, context=context)
            outer.attention_auto_machine = AttentionAutoMachine.Read
            outer.attention_auto_machine_weight = config.reference_weight

        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None: h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

    def hacked_basic_transformer_inner_forward(self, x, context=None):
        x_norm1 = self.norm1(x)
        self_attn1 = 0
        if self.disable_self_attn:
            # Do not use self-attention
            self_attn1 = self.attn1(x_norm1, context=context)
        else:
            # Use self-attention
            self_attention_context = x_norm1
            if outer.attention_auto_machine == AttentionAutoMachine.Write:
                self.bank.append(self_attention_context.detach().clone())
            if outer.attention_auto_machine == AttentionAutoMachine.Read:
                if outer.attention_auto_machine_weight > self.attn_weight:
                    self_attention_context = torch.cat([self_attention_context] + self.bank, dim=1)
                self.bank.clear()
            self_attn1 = self.attn1(x_norm1, context=self_attention_context)

        x = self_attn1 + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

    if config.reference_active:
        # outer = sd_model.model.diffusion_model
        try:
            outer.forward = outer.original_forward
        except:
            pass
        outer.original_forward = outer.forward
        outer.attention_auto_machine_weight = config.reference_weight
        outer.forward = control_forward
        outer.attention_auto_machine = AttentionAutoMachine.Read
        print('Using reference control.')

        attn_modules = [module for module in torch_dfs(outer) if isinstance(module, BasicTransformerBlock)]
        attn_modules = sorted(attn_modules, key=lambda x: - x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            module._original_inner_forward = module._forward
            module._forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = []
            module.attn_weight = float(i) / float(len(attn_modules))

        outer.attn_module_list = attn_modules
        for module in outer.attn_module_list:
            module.bank = []
