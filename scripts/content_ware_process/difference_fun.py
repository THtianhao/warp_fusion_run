#@title Create schedules from frame difference
import numpy as np

from scripts.run.run_common_func import get_scheduled_arg

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


