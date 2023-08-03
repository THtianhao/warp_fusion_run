import scripts.settings.setting
from scripts.utils.env import root_dir

print(root_dir)
import pydevd_pycharm

pydevd_pycharm.settrace('49.7.62.197', port=10090, stdoutToServer=True, stderrToServer=True)

from scripts.lora_embedding.lora_and_embedding_ import set_lora_embedding
from scripts.lora_embedding.lora_embedding_config import LoraEmbeddingConfig
from scripts.refrerence_control_processor.reference_config import ReferenceConfig
from scripts.refrerence_control_processor.reference_control import reference_control
from scripts.run.prepare_run import prepare_run
from scripts.run.run_func import do_run
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
from scripts.video_process.video_flow import extra_video_frame, extra_background_mask, download_reference_repository, set_video_path
from scripts.video_process.generate_optical_func import generate_optical_flow
from scripts.settings.main_config import MainConfig
from scripts.run.run_prepare_config import run_prepare_config

if __name__ == "__main__":
    main_config = MainConfig()
    main_config.check_consistency = False  # 不检查光流一致性
    main_config.fixed_seed = True
    main_config.warp_forward = False  # 进行光流融合
    main_config.frame_range = [0, 0]
    main_config.steps_schedule ={0: 0, 148: 25}

    # 使用红色背景
    main_config.use_background_mask = True
    main_config.background = 'init_video'
    main_config.background_source = 'red'
    # main_config.text_prompts = {0: ['Masterpiece, beautiful white marble statue']}
    main_config.text_prompts = {0: ['masterpiece, best quality, 1man, sword, glowing']}

    # main_config.masked_guidance = True # 测试10和11用到
    # main_config.cc_masked_diffusion = 0.7
    main_config.alpha_masked_diffusion = 0.5 # 测试12用到

    video_config = VideoConfig()
    # video_config.video_init_path = "./res/dance.mp4"
    # video_config.video_init_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"
    video_config.video_init_path = "/data/tianhao/warp_fussion/video/wuxia.mp4"
    # Video Mask Setting
    video_config.mask_source = 'init_video'
    video_config.extract_background_mask = True
    # video_config.mask_video_path = "/data/tianhao/jupyter-notebook/warpfusion/video/dance.mp4"

    set_video_path(video_config)
    extra_video_frame(video_config)
    extra_background_mask(video_config)

    download_reference_repository(video_config.animation_mode)
    # 使用光流脚本生成光流图，生成一致性图
    video_config.flow_warp = False  # 使用光流
    video_config.use_jit_raft = False  # 使用torch jit的版本，不适用torch2.0
    video_config.force_flow_generation = True
    video_config.flow_save_img_preview = False  # 是否生成用户可读的光流图
    video_config.flow_lq = True  # 使用半精度
    generate_optical_flow(video_config)

    model_config = ModelConfig()
    model_config.force_download = False
    model_config.model_path = '/data/tianhao/stable-diffusion-webui/models/Stable-diffusion/ChosenChineseStyleNsfw_v10.ckpt'
    model_config.controlnet_models_dir = '/data/tianhao/warp_fussion/ControlNet/models'
    load_sd_and_k_fusion(model_config, main_config)
    # 禁用tail_vae
    tail_vae_config = TiledVaeConfig()
    tail_vae_config.use_tiled_vae = False
    tiled_vae(tail_vae_config, model_config.sd_model)
    # 加载图片相似度clip模型
    clip_config = ClipConfig()
    clip_config.clip_guidance_scale = 0
    get_clip_model_size(clip_config)

    content_aware_config = ContentAwareConfig()
    content_aware(content_aware_config, video_config.videoFramesFolder)

    captioning_config = CaptioningConfig()
    generate_key_frame(captioning_config, video_config.videoFramesFolder, content_aware_config.diff)

    lora_embedding_config = LoraEmbeddingConfig()
    lora_embedding_config.lora_dir = '/data/tianhao/warp_fussion/models/loras'
    lora_embedding_config.custom_embed_dir = '/data/tianhao/warp_fussion/models/embeddings'
    set_lora_embedding(lora_embedding_config)

    # controlnet 1.1 ReferenceControl 的实现,图生图
    ref_config = ReferenceConfig()
    # ref_config.use_reference = True
    reference_control(ref_config, model_config.sd_model, main_config.reference_latent)

    prepare_run(main_config)
    run_prepare_config(main_config, model_config, video_config, lora_embedding_config, content_aware_config)
    do_run(main_config, video_config, content_aware_config, model_config, ref_config, captioning_config, clip_config)
