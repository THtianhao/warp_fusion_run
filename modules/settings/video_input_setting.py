import subprocess
import sys

from PIL.Image import Image

from modules.settings.setting import batchFolder
from modules.utils.env import root_dir
from modules.utils.path import createPath

animation_mode = 'Video Input'
import os, platform
if platform.system() != 'Linux' and not os.path.exists("ffmpeg.exe"):
  print("Warning! ffmpeg.exe not found. Please download ffmpeg and place it in current working dir.")


#@markdown ---


video_init_path = "d:/init_images/Snapinsta.app_video_314B712AC1DB9308A6E8D7CCFB65EF8C_video_dashinit.mp4" #@param {type: 'string'}

extract_nth_frame =  1#@param {type: 'number'}
#@markdown *Specify frame range. end_frame=0 means fill the end of video*
start_frame = 0#@param {type: 'number'}
end_frame = 0#@param {type: 'number'}
end_frame_orig = end_frame
if end_frame<=0 or end_frame==None: end_frame = 99999999999999999999999999999
#@markdown ####Separate guiding video (optical flow source):
#@markdown Leave blank to use the first video.
flow_video_init_path = "" #@param {type: 'string'}
flow_extract_nth_frame =  1#@param {type: 'number'}
if flow_video_init_path == '':
  flow_video_init_path = None
#@markdown ####Image Conditioning Video Source:
#@markdown Used together with image-conditioned models, like controlnet, depth, or inpainting model.
#@markdown You can use your own video as depth mask or as inpaiting mask.
cond_video_path = "" #@param {type: 'string'}
cond_extract_nth_frame =  1#@param {type: 'number'}
if cond_video_path == '':
  cond_video_path = None

#@markdown ####Colormatching Video Source:
#@markdown Used as colormatching source. Specify image or video.
color_video_path = "" #@param {type: 'string'}
color_extract_nth_frame =  1#@param {type: 'number'}
if color_video_path == '':
  color_video_path = None
#@markdown Enable to store frames, flow maps, alpha maps on drive
store_frames_on_google_drive = False #@param {type: 'boolean'}
video_init_seed_continuity = False

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
  createPath(output_path)
  print(f"Exporting Video Frames (1 every {nth_frame})...")
  try:
    for f in [o.replace('\\','/') for o in glob(output_path+'/*.jpg')]:
    # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
      pathlib.Path(f).unlink()
  except:
    print('error deleting frame ', f)
  # vf = f'select=not(mod(n\\,{nth_frame}))'
  vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
  if os.path.exists(video_path):
    try:
        subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    except:
        subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'], stdout=subprocess.PIPE).stdout.decode('utf-8')

  else:
    sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')

if animation_mode == 'Video Input':
  postfix = f'{generate_file_hash(video_init_path)[:10]}_{start_frame}_{end_frame_orig}_{extract_nth_frame}'
  if flow_video_init_path:
    flow_postfix = f'{generate_file_hash(flow_video_init_path)[:10]}_{flow_extract_nth_frame}'
  if store_frames_on_google_drive: #suggested by Chris the Wizard#8082 at discord
      videoFramesFolder = f'{batchFolder}/videoFrames/{postfix}'
      flowVideoFramesFolder = f'{batchFolder}/flowVideoFrames/{flow_postfix}' if flow_video_init_path else videoFramesFolder
      condVideoFramesFolder = f'{batchFolder}/condVideoFrames'
      colorVideoFramesFolder = f'{batchFolder}/colorVideoFrames'
      controlnetDebugFolder = f'{batchFolder}/controlnetDebug'
      recNoiseCacheFolder = f'{batchFolder}/recNoiseCache'

  else:
      videoFramesFolder = f'{root_dir}/videoFrames/{postfix}'
      flowVideoFramesFolder = f'{root_dir}/flowVideoFrames/{flow_postfix}' if flow_video_init_path else videoFramesFolder
      condVideoFramesFolder = f'{root_dir}/condVideoFrames'
      colorVideoFramesFolder = f'{root_dir}/colorVideoFrames'
      controlnetDebugFolder = f'{root_dir}/controlnetDebug'
      recNoiseCacheFolder = f'{root_dir}/recNoiseCache'

  os.makedirs(controlnetDebugFolder, exist_ok=True)
  os.makedirs(recNoiseCacheFolder, exist_ok=True)

  extractFrames(video_init_path, videoFramesFolder, extract_nth_frame, start_frame, end_frame)
  if flow_video_init_path:
    print(flow_video_init_path, flowVideoFramesFolder, flow_extract_nth_frame)
    extractFrames(flow_video_init_path, flowVideoFramesFolder, flow_extract_nth_frame, start_frame, end_frame)

  if cond_video_path:
    print(cond_video_path, condVideoFramesFolder, cond_extract_nth_frame)
    extractFrames(cond_video_path, condVideoFramesFolder, cond_extract_nth_frame, start_frame, end_frame)

  if color_video_path:
    try:
      os.makedirs(colorVideoFramesFolder, exist_ok=True)
      Image.open(color_video_path).save(os.path.join(colorVideoFramesFolder,'000001.jpg'))
    except:
      print(color_video_path, colorVideoFramesFolder, color_extract_nth_frame)
      extractFrames(color_video_path, colorVideoFramesFolder, color_extract_nth_frame, start_frame, end_frame)