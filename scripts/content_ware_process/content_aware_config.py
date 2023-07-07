from dataclasses import dataclass
@dataclass
class ContentAwareConfig:
    analyze_video = False  # @param {'type':'boolean'}
    diff_function = 'lpips'  # @param ['rmse','lpips','rmse+lpips']
