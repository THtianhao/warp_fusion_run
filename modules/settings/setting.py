from modules.utils.path import createPath
from modules.utils.env import outDirPath

batch_name = 'stable_warpfusion_0.15.0'  #@param{type: 'string'}
steps = 50
##@param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
# stop_early = 0  #@param{type: 'number'}
stop_early = 0
stop_early = min(steps - 1, stop_early)
#@markdown Specify desired output size here.\
#@markdown Don't forget to rerun all steps after changing the width height (including forcing optical flow generation)
width_height = [720, 1280]  #@param{type: 'raw'}
width_height = [int(o) for o in width_height]
clip_guidance_scale = 0  #
tv_scale = 0
range_scale = 0
cutn_batches = 4
skip_augs = False

#@markdown ---

#@markdown ####**Init Settings:**
init_image = ""  #@param{type: 'string'}
init_scale = 0
##@param{type: 'integer'}
skip_steps = 25
##@param{type: 'integer'}
##@markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.\
##@markdown A good init_scale for Stable Diffusion is 0*

#Get corrected sizes
#@markdown Make sure the resolution is divisible by that number. The Default 64 is the most stable.

force_multiple_of = "64"  #@param [8,64]
force_multiple_of = int(force_multiple_of)
side_x = (width_height[0] // force_multiple_of) * force_multiple_of
side_y = (width_height[1] // force_multiple_of) * force_multiple_of
if side_x != width_height[0] or side_y != width_height[1]:
    print(f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of {force_multiple_of}.')
width_height = (side_x, side_y)
#Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps

#Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)