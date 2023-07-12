import os
import sys
import pydevd_pycharm

from scripts.settings.main_config import MainConfig

pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)
from scripts.utils.env import root_dir

print(root_dir)

# from scripts.captioning_process.captioning_config import CaptioningConfig
# from scripts.captioning_process.generate_key_frame import generate_key_frame
# from scripts.content_ware_process.content_aware_config import ContentAwareConfig
# from scripts.content_ware_process.content_aware_scheduing import content_aware
from scripts.video_process.video_config import VideoConfig
# from scripts.clip_process.clip_config import ClipConfig
# from scripts.clip_process.clip_process import get_clip_model_size
from scripts.model_process.load_sd_k_model import load_sd_and_k_fusion
from scripts.model_process.model_config import ModelConfig
# from scripts.tiled_vae_process.tiled_vae import tiled_vae
# from scripts.tiled_vae_process.tiled_vae_config import TiledVaeConfig
from scripts.video_process.video_flow import extra_video_frame, mask_video_frame, download_reference_repository, set_video_path
from scripts.video_process.generate_optical_func import generate_optical_flow
if __name__ == "__main__":
    main_config = MainConfig()
    video_config = VideoConfig()
    # video_config.video_init_path = "./res/dance.mp4"
    video_config.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    set_video_path(video_config)
    # extra_video_frame(video_config)
    video_config.extract_background_mask = False
    video_config.mask_source = 'init_video'
    video_config.mask_video_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance_mask.mp4"
    mask_video_frame(video_config)


    # download_reference_repository(video_config.animation_mode)
    # 使用光流脚本生成光流图，生成一致性图
    video_config.use_jit_raft = False
    generate_optical_flow(video_config)
    model_config = ModelConfig()
    model_config.model_path = '/data/tianhao/stable-diffusion-webui/models/Stable-diffusion/deliberate_v2.safetensors'
    model_config.controlnet_models_dir = '/data/tianhao/warp_fussion/ControlNet/models'
    load_sd_and_k_fusion(model_config,main_config)
    # tail_vae_config = TiledVaeConfig()
    # tiled_vae(tail_vae_config, model_config.sd_mode)
    # clip_config = ClipConfig()
    # get_clip_model_size(clip_config)
    # content_aware_config = ContentAwareConfig()
    # content_aware(content_aware_config, video_config.videoFramesFolder)
    # captioning_config = CaptioningConfig()
    # generate_key_frame(captioning_config, video_config.videoFramesFolder)
