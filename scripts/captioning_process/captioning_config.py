from dataclasses import dataclass

@dataclass
class CaptioningConfig:
    make_captions = False  # @param {'type':'boolean'}
    keyframe_source = 'Every n-th frame'  # @param ['Content-aware scheduling keyframes', 'User-defined keyframe list', 'Every n-th frame']
    # @markdown This option only works with  keyframe source == User-defined keyframe list
    user_defined_keyframes = [3, 4, 5]  # @param
    # @markdown This option only works with  keyframe source == Content-aware scheduling keyframes
    diff_thresh = 0.33  # @param {'type':'number'}
    # @markdown This option only works with  keyframe source == Every n-th frame
    nth_frame = 60  # @param {'type':'number'}
