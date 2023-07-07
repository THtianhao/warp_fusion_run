from dataclasses import dataclass

@dataclass

class ReferenceConfig:
    use_reference = False  # @param {'type':'boolean'}
    reference_weight = 0.5  # @param
    reference_source = 'init'  # @param ['stylized', 'init', 'prev_frame','color_video']
    reference_mode = 'Balanced'  # @param ['Balanced', 'Controlnet', 'Prompt']
