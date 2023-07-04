import subprocess
import os
from utils.cmd import pipi, pipie, pipis, gitclone, unpip

from utils.nvidia_utils import print_card

if __name__ == "__main__":
    print_card()
    pipis(['tqdm','ipwidgets==7.7.1','protobuf==3.20.3'])

    from tqdm.notebook import tqdm
    progress_bar = tqdm(total= 25)
    progress_bar.set_description("Install dependencies")
    pipis(['mediapipe','piexif','safetensors','lark'])
    unpip('torchtext')
    gitclone('https://github.com/sxela/sxela-stablediffusion',
             dest='stablediffusion')
    gitclone('https://github.com/sxela/controlnet-v1-1-nightly',
             dest='controlnet')
    gitclone('https://github.com/pengbo-learn/python-color-transfer')
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
    progress_bar.update(2)
    pipi('Pillow==9.0.0')
    pipie('./stablediffusion')
    progress_bar.update(2)
    pipis(['ipywidgets==7.7.1','transformers==4.19.2','omegaconf','einops','pytorch_lightning>1.4.11,<=1.7.7', 'scikit-image','opencv-python'])
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
    progress_bar.update(2)
