import os

from modules.bean.video_bean import VideoBean
from modules.video_masking import video_setting, mask_video

if __name__ == "__main__":
    bean = VideoBean()
    # bean.video_init_path = "./res/dance.mp4"
    bean.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    video_setting(bean)
    # mask_video(bean)

