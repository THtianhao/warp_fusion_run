from dataclasses import dataclass

@dataclass
class ContentAwareConfig:
    analyze_video = False  # @param {'type':'boolean'}
    diff_function = 'lpips'  # @param ['rmse','lpips','rmse+lpips']
    diff = [0]
    # @markdown fill in templates for schedules you'd like to create from frames' difference\
    # @markdown leave blank to use schedules from previous cells\
    # @markdown format: **[normal value, high difference value, difference threshold, falloff from high to normal (number of frames)]**\
    # @markdown For example, setting flow blend template to [0.999, 0.3, 0.5, 5] will use 0.999 everywhere unless a scene has changed (frame difference >0.5) and then set flow_blend for this frame to 0.3 and gradually fade to 0.999 in 5 frames

    latent_scale_template = ''  # @param {'type':'raw'}
    init_scale_template = ''  # @param {'type':'raw'}
    steps_template = ''  # @param {'type':'raw'}
    style_strength_template = [0.8, 0.8, 0.5, 5]  # @param {'type':'raw'}
    flow_blend_template = [1, 0., 0.5, 2]  # @param {'type':'raw'}
    cfg_scale_template = None  # @param {'type':'raw'}
    image_scale_template = None  # @param {'type':'raw'}

    # @markdown Turning this off will disable templates and will use schedules set in previous cell
    make_schedules = False  # @param {'type':'boolean'}
    # @markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
    respect_sched = True  # @param {'type':'boolean'}
    diff_override = []  # @param {'type':'raw'}
    # shift+1 required
