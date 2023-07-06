import subprocess
import sys
import hashlib
import pathlib
from glob import glob
from scripts.utils.path import createPath
import os

def extractFrames(video_path, output_path, nth_frame, start_frame, end_frame):
    createPath(output_path)
    print(f"Exporting Video Frames (1 every {nth_frame})...")
    try:
        for f in [o.replace('\\', '/') for o in glob(output_path + '/*.jpg')]:
            # for f in pathlib.Path(f'{output_path}').glob('*.jpg'):
            pathlib.Path(f).unlink()
    except:
        print('error deleting frame ', f)
    # vf = f'select=not(mod(n\\,{nth_frame}))'
    vf = f'select=between(n\\,{start_frame}\\,{end_frame}) , select=not(mod(n\\,{nth_frame}))'
    if os.path.exists(video_path):
        try:
            subprocess.run(['ffmpeg', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'],
                           stdout=subprocess.PIPE).stdout.decode('utf-8')
        except:
            subprocess.run(['ffmpeg.exe', '-i', f'{video_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '2', '-loglevel', 'error', '-stats', f'{output_path}/%06d.jpg'],
                           stdout=subprocess.PIPE).stdout.decode('utf-8')

    else:
        sys.exit(f'\nERROR!\n\nVideo not found: {video_path}.\nPlease check your video path.\n')

def generate_file_hash(input_file):
    # Get file name and metadata
    file_name = os.path.basename(input_file)
    file_size = os.path.getsize(input_file)
    creation_time = os.path.getctime(input_file)
    # Generate hash
    hasher = hashlib.sha256()
    hasher.update(file_name.encode('utf-8'))
    hasher.update(str(file_size).encode('utf-8'))
    hasher.update(str(creation_time).encode('utf-8'))
    file_hash = hasher.hexdigest()
    return file_hash
