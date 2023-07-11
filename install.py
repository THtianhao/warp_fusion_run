import subprocess
import os
import sys
from scripts.utils.cmd import install_requirement, pipi, pipie, pipis, gitclone, unpip
from scripts.utils.nvidia_utils import print_card
from scripts.utils.env import root_path

if __name__ == "__main__":
    print_card()
    pipis(['ipwidgets==7.7.1', 'protobuf==3.20.3'])
    pipis(['mediapipe', 'piexif', 'safetensors', 'lark'])
    unpip('torchtext')
    gitclone('https://github.com/sxela/sxela-stablediffusion',
             dest='stablediffusion')
    gitclone('https://github.com/sxela/controlnet-v1-1-nightly',
             dest='ControlNet')
    gitclone('https://github.com/pengbo-learn/python-color-transfer')
    gitclone('https://github.com/Sxela/k-diffusion')
    try:
        if os.path.exists('./stablediffusion'):
            print('pulling a fresh stablediffusion')
            os.chdir(f'./stablediffusion')
            subprocess.run(['git', 'pull'])
            os.chdir(f'../')
    except:
        pass
    try:
        if os.path.exists('./controlnet'):
            print('pulling a fresh controlnet')
            os.chdir(f'./controlnet')
            subprocess.run(['git', 'pull'])
            os.chdir(f'../')
    except:
        pass
    pipie('./stablediffusion')
    try:
        if os.path.exists('./k-diffusion'):
            print('pulling a fresh k-diffusion')
            os.chdir(f'./k-diffusion')
            subprocess.run(['git', 'pull'])
            pipie('.')
            os.chdir(f'../')
    except:
        pass
    pipie('./k-diffusion')
    pipis([
        'Pillow==9.0.0',
        'ipywidgets==7.7.1',
        'transformers==4.19.2',
        'omegaconf',
        'einops',
        # 'pytorch_lightning>1.4.11,<=1.7.7',
        'pytorch_lightning>1.4.11',
        'scikit-image',
        'opencv-python',
        'ai-tools',
        'cognitive-face',
        'zprint',
        'kornia==0.5.0',
    ])
    pipie('git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers')
    pipie('git+https://github.com/openai/CLIP.git@main#egg=clip')
    pipis([
        'lpips',
        'keras',
    ])
    sys.path.append('./k-diffusion')
    pipis([
        'wget', 'webdataset', 'open_clip_torch', 'opencv-python==4.5.5.64',
        'pandas', 'matplotlib', 'fvcore', 'lpips', 'datetime', 'timm==0.6.13',
        'ftfy', 'einops', 'pytorch-lightning', 'omegaconf', 'prettytable',
        'fairscale'
    ])
    PROJECT_DIR = os.path.abspath(os.getcwd())
    if not os.path.exists('guided-diffusion'):
        gitclone("https://github.com/crowsonkb/guided-diffusion")
    sys.path.append(f'{PROJECT_DIR}/guided-diffusion')
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f'{PROJECT_DIR}/ResizeRight')
    if not os.path.exists("BLIP"):
        gitclone("https://github.com/salesforce/BLIP")
    sys.path.append(f'{PROJECT_DIR}/BLIP')
    os.chdir(root_path)
    gitclone('https://github.com/xinntao/Real-ESRGAN')
    os.chdir('Real-ESRGAN')
    pipis(['basicsr', 'google-cloud-vision', 'ffmpeg'])
    print("install requirement")
    install_requirement()
    print('run setup.py')
    res = subprocess.run(['python', 'setup.py', 'develop'],
                         stdout=subprocess.PIPE).stdout.decode('utf-8')
    os.chdir(root_path)
    print("Done")
