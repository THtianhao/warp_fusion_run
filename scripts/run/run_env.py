stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0 / 200.
VERBOSE = True
blend_json_schedules = False
diffusion_model = "stable_diffusion"
diffusion_sampling_mode = 'ddim'
loaded_controlnets = {}