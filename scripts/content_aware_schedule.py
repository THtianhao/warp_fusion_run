#@title Content-aware scheduing
#@markdown Allows automated settings scheduling based on video frames difference. If a scene changes, it will be detected and reflected in the schedule.\
#@markdown rmse function is faster than lpips, but less precise.\
#@markdown After the analysis is done, check the graph and pick a threshold that works best for your video. 0.5 is a good one for lpips, 1.2 is a good one for rmse. Don't forget to adjust the templates with new threshold in the cell below.

def load_img_lpips(path, size=(512,512)):
    image = Image.open(path).convert("RGB")
    image = image.resize(size, resample=Image.LANCZOS)
    # print(f'resized to {image.size}')
    image = np.array(image).astype(np.float32) / 127
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = normalize(image)
    return image.cuda()

diff = None
analyze_video = False #@param {'type':'boolean'}

diff_function = 'lpips' #@param ['rmse','lpips','rmse+lpips']

def l1_loss(x,y):
  return torch.sqrt(torch.mean((x-y)**2))


def rmse(x,y):
  return torch.abs(torch.mean(x-y))

def joint_loss(x,y):
  return rmse(x,y)*lpips_model(x,y)

diff_func = rmse
if  diff_function == 'lpips':
  diff_func = lpips_model
if diff_function == 'rmse+lpips':
  diff_func = joint_loss

if analyze_video:
  diff = [0]
  frames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
  from tqdm.notebook import trange
  for i in trange(1,len(frames)):
    with torch.no_grad():
      diff.append(diff_func(load_img_lpips(frames[i-1]), load_img_lpips(frames[i])).sum().mean().detach().cpu().numpy())

  import numpy as np
  import matplotlib.pyplot as plt

  plt.rcParams["figure.figsize"] = [12.50, 3.50]
  plt.rcParams["figure.autolayout"] = True

  y = diff
  plt.title(f"{diff_function} frame difference")
  plt.plot(y, color="red")
  calc_thresh = np.percentile(np.array(diff), 97)
  plt.axhline(y=calc_thresh, color='b', linestyle='dashed')

  plt.show()
  print(f'suggested threshold: {calc_thresh.round(2)}')

  # @title Plot threshold vs frame difference
  # @markdown The suggested threshold may be incorrect, so you can plot your value and see if it covers the peaks.
  if diff is not None:
      import numpy as np
      import matplotlib.pyplot as plt

      plt.rcParams["figure.figsize"] = [12.50, 3.50]
      plt.rcParams["figure.autolayout"] = True

      y = diff
      plt.title(f"{diff_function} frame difference")
      plt.plot(y, color="red")
      calc_thresh = np.percentile(np.array(diff), 97)
      plt.axhline(y=calc_thresh, color='b', linestyle='dashed')
      user_threshold = 0.13  # @param {'type':'raw'}
      plt.axhline(y=user_threshold, color='r')

      plt.show()
      peaks = []
      for i, d in enumerate(diff):
          if d > user_threshold:
              peaks.append(i)
      print(f'Peaks at frames: {peaks} for user_threshold of {user_threshold}')
  else:
      print('Please analyze frames in the previous cell  to plot graph')

, threshold
#@title Create schedules from frame difference
def adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched=None):
  diff_array = np.array(diff)

  diff_new = np.zeros_like(diff_array)
  diff_new = diff_new+normal_val

  for i in range(len(diff_new)):
    el = diff_array[i]
    if sched is not None:
      diff_new[i] = get_scheduled_arg(i, sched)
    if el>thresh or i==0:
      diff_new[i] = new_scene_val
      if falloff_frames>0:
        for j in range(falloff_frames):
          if i+j>len(diff_new)-1: break
          # print(j,(falloff_frames-j)/falloff_frames, j/falloff_frames )
          falloff_val = normal_val
          if sched is not None:
            falloff_val = get_scheduled_arg(i+falloff_frames, sched)
          diff_new[i+j] = new_scene_val*(falloff_frames-j)/falloff_frames+falloff_val*j/falloff_frames
  return diff_new

def check_and_adjust_sched(sched, template, diff, respect_sched=True):
  if template is None or template == '' or template == []:
    return sched
  normal_val, new_scene_val, thresh, falloff_frames = template
  sched_source = None
  if respect_sched:
    sched_source = sched
  return list(adjust_schedule(diff, normal_val, new_scene_val, thresh, falloff_frames, sched_source).astype('float').round(3))

#@markdown fill in templates for schedules you'd like to create from frames' difference\
#@markdown leave blank to use schedules from previous cells\
#@markdown format: **[normal value, high difference value, difference threshold, falloff from high to normal (number of frames)]**\
#@markdown For example, setting flow blend template to [0.999, 0.3, 0.5, 5] will use 0.999 everywhere unless a scene has changed (frame difference >0.5) and then set flow_blend for this frame to 0.3 and gradually fade to 0.999 in 5 frames

latent_scale_template = '' #@param {'type':'raw'}
init_scale_template = '' #@param {'type':'raw'}
steps_template = '' #@param {'type':'raw'}
style_strength_template = [0.8, 0.8, 0.5, 5] #@param {'type':'raw'}
flow_blend_template = [1, 0., 0.5, 2] #@param {'type':'raw'}
cfg_scale_template = None #@param {'type':'raw'}
image_scale_template = None #@param {'type':'raw'}

#@markdown Turning this off will disable templates and will use schedules set in previous cell
make_schedules = False #@param {'type':'boolean'}
#@markdown Turning this on will respect previously set schedules and only alter the frames with peak difference
respect_sched = True #@param {'type':'boolean'}
diff_override = [] #@param {'type':'raw'}

#shift+1 required
