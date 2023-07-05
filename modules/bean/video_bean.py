from dataclasses import dataclass
@dataclass
class VideoBean:
    animation_mode = 'Video Input'
    video_init_path = ""
    extract_nth_frame = 1  # @param {type: 'number'}
    # *Specify frame range. end_frame=0 means fill the end of video*
    start_frame = 0  # @param {type: 'number'}
    end_frame = 0  # @param {type: 'number'}
    end_frame_orig = end_frame
    if end_frame <= 0 or end_frame == None: end_frame = 99999999999999999999999999999
    #  ####Separate guiding video (optical flow source):
    #  Leave blank to use the first video.
    flow_video_init_path = ""  # @param {type: 'string'}
    flow_extract_nth_frame = 1  # @param {type: 'number'}
    if flow_video_init_path == '':
        flow_video_init_path = None
    #  ####Image Conditioning Video Source:
    #  Used together with image-conditioned models, like controlnet, depth, or inpainting model.
    #  You can use your own video as depth mask or as inpaiting mask.
    cond_video_path = ""  # @param {type: 'string'}
    cond_extract_nth_frame = 1  # @param {type: 'number'}
    if cond_video_path == '':
        cond_video_path = None
    #  ####Colormatching Video Source:
    #  Used as colormatching source. Specify image or video.
    color_video_path = ""  # @param {type: 'string'}
    color_extract_nth_frame = 1  # @param {type: 'number'}
    if color_video_path == '':
        color_video_path = None
    #  Enable to store frames, flow maps, alpha maps on drive
    store_frames_on_google_drive = False  # @param {type: 'boolean'}
    video_init_seed_continuity = False
    videoFramesFolder = ''