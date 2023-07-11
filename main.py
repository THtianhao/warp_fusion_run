import os
import sys
from scripts.utils.env import root_dir

print(root_dir)

from scripts.captioning_process.captioning_config import CaptioningConfig
from scripts.captioning_process.generate_key_frame import generate_key_frame
from scripts.content_ware_process.content_aware_config import ContentAwareConfig
from scripts.content_ware_process.content_aware_scheduing import content_aware
from scripts.video_process.video_config import VideoConfig
from scripts.clip_process.clip_config import ClipConfig
from scripts.clip_process.clip_process import get_clip_model_size
from scripts.model_process.load_sd_k_model import load_sd_and_k_fusion
from scripts.model_process.model_config import ModelConfig
from scripts.tiled_vae_process.tiled_vae import tiled_vae
from scripts.tiled_vae_process.tiled_vae_config import TiledVaeConfig
from scripts.video_process.video_flow import extra_video_frame, mask_video, download_reference_repository
from scripts.video_process.generate_optical_func import generate_optical_flow

if __name__ == "__main__":
    video_config = VideoConfig()
    # video_config.video_init_path = "./res/dance.mp4"
    video_config.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    extra_video_frame(video_config)
    # mask_video(video_config)
    # download_reference_repository(video_config.animation_mode)
    # generate_optical_flow(video_config)
    # model_config = ()
    # load_sd_and_k_fusion(model_config)
    # tail_vae_config = TiledVaeConfig()
    # tiled_vae(tail_vae_config, model_config.sd_mode)
    # clip_config = ClipConfig()
    # get_clip_model_size(clip_config)
    # content_aware_config = ContentAwareConfig()
    # content_aware(content_aware_config, video_config.videoFramesFolder)
    # captioning_config = CaptioningConfig()
    # generate_key_frame(captioning_config, video_config.videoFramesFolder)
