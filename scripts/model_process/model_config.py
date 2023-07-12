from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_path = "d:/models/revAnimated_v122.safetensors"  # @param {'type':'string'}
    #  ControlNet download settings
    #  ControlNet downloads are managed by controlnet_multi settings in Main settings tab.
    use_small_controlnet = True
    # #@param {'type':'boolean'}
    small_controlnet_model_path = ''
    # #@param {'type':'string'}
    download_control_model = True
    # #@param {'type':'boolean'}
    force_download = False  # @param {'type':'boolean'}
    controlnet_models_dir = "d:/models/ControlNet"  # @param {'type':'string'}
    img_zero_uncond = False  # by default image conditioned models use same image for negative conditioning (i.e. both positive and negative image conditings are the same. you can use empty negative condition by enabling this)
    sd_mode = None
    model_wrap_cfg = None
    sigma_min = None
    sigma_max = None
    model_wrap = None
