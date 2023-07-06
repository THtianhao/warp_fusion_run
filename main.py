from scripts.bean.video_config_bean import VideoConfigBean
from scripts.model_process.load_sd_k_model import load_sd_and_k_fusion
from scripts.model_process.model_config import ModelConfig
from scripts.video_flow import extra_video_frame, mask_video, download_reference_repository
from scripts.video_process.generate_optical_func import generate_optical_flow

if __name__ == "__main__":
    bean = VideoConfigBean()
    bean.video_init_path = "./res/dance.mp4"
    # bean.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    extra_video_frame(bean)
    mask_video(bean)
    download_reference_repository(bean.animation_mode)
    generate_optical_flow(bean)
    model_config = ModelConfig()
    load_sd_and_k_fusion(model_config)
