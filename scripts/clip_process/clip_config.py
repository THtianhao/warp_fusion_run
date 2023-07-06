from dataclasses import dataclass
@dataclass
class ClipConfig:
    clip_guidance_scale = 0  # @param {'type':"number"}
    clip_model = None
    clip_size = None
