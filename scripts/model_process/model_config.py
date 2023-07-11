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
    sd_mode = None
    model_wrap_cfg = None
    sigma_min = None
    sigma_max = None
