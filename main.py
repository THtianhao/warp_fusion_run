import os

from modules.bean.video_bean import VideoBean
from modules.video_masking import video_setting, mask_video

if __name__ == "__main__":
    bean = VideoBean()
    video_setting(bean)
    mask_video(bean)

