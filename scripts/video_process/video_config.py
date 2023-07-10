from dataclasses import dataclass

@dataclass
class VideoConfig:
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
    flowVideoFramesFolder = ''
    condVideoFramesFolder = ''
    colorVideoFramesFolder = ''
    controlnetDebugFolder = ''
    recNoiseCacheFolder = ''
    videoFramesAlpha = ''
    flo_folder = ''
    temp_flo = ''
    flo_fwd_folder = ''
    flo_bck_folder = ''
    in_path = ''

    flow_warp = True
    check_consistency = True
    force_flow_generation = False  # @param {type:'boolean'}

    use_legacy_cc = False  # @param{'type':'boolean'}
    threads = 4  # @param {'type':'number'}
    # @markdown If you're having "process died" error on Windows, set num_workers to 0
    num_workers = 0  # @param {'type':'number'}

    # @markdown Use lower quality model (half-precision).\
    # @markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
    flow_lq = True  # @param {type:'boolean'}
    # @markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
    flow_save_img_preview = False  # @param {type:'boolean'}

    # #@markdown reverse_cc_order - on - default value (like in older notebooks). off - reverses consistency computation
    reverse_cc_order = True  #
    # #@param {type:'boolean'}
    if not flow_warp: print('flow_wapr not set, skipping')
    # @markdown Use previous pre-compile raft version (won't work with pytorch 2.0)
    use_jit_raft = False  # @param {'type':'boolean'}
    # @markdown Compile raft model (only with use_raft_jit = False). Compiles the model (~about 2 minutes) for ~30% speedup. Use for very long runs.
    compile_raft = False  # @param {'type':'boolean'}
    # @markdown Flow estimation quality (number of iterations, 12 - default. higher - better and slower)
    num_flow_updates = 12  # @param {'type':'number'}
    # \@markdown Unreliable areas mask (missed consistency) width
    # \@markdown Default = 1
    missed_consistency_dilation = 2  # \ @param {'type':'number'}
    # \@markdown Motion edge areas (edge consistency) width
    # \@markdown Default = 11
    edge_consistency_width = 11  # \@param {'type':'number'}

