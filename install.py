import subprocess
from utils.cmd import pipis

from utils.nvidia_utils import print_card



if __name__ == "__main__":
    print_card()
    pipis(['tqdm', 'ipywidgets==7.7.1','protobuf==3.20.3'])
    progress_bar = tqdm(total=52)
    progress_bar.set_description("Installing dependencies")
    with io.capture_output(stderr=False) as captured:
        pipis(['mediapipe','piexif','safetensors','lark'])
        !python -m pip -q uninstall torchtext -y
        progress_bar.update(3) #10
        gitclone('https://github.com/Sxela/sxela-stablediffusion', dest = 'stablediffusion')
        # !git clone -b sdp-attn https://github.com/Sxela/sxela-stablediffusion stablediffusion
        gitclone('https://github.com/Sxela/ControlNet-v1-1-nightly', dest = 'ControlNet')
        gitclone('https://github.com/pengbo-learn/python-color-transfer')
        progress_bar.update(3) #20
        try:
            if os.path.exists('./stablediffusion'):
                print('pulling a fresh stablediffusion')
            os.chdir( f'./stablediffusion')
            subprocess.run(['git', 'pull'])
            os.chdir( f'../')
        except:
            pass
        try:
            if os.path.exists('./ControlNet'):
                print('pulling a fresh ControlNet')
                os.chdir( f'./ControlNet')
                subprocess.run(['git', 'pull'])
                os.chdir( f'../')
        except:
            pass
        progress_bar.update(2) #25
        !python -m pip -q install --ignore-installed Pillow==9.0.0
        !python -m pip -q install -e ./stablediffusion
        progress_bar.update(2)
        !python -m pip -q install ipywidgets==7.7.1
        !python -m pip -q install transformers==4.19.2
        progress_bar.update(2)
        !python -m pip -q install omegaconf
        !python -m pip -q install einops
        !python -m pip -q install "pytorch_lightning>1.4.1,<=1.7.7"
        progress_bar.update(3) #30
        !python -m pip -q install scikit-image
        !python -m pip -q install opencv-python
        progress_bar.update(2)
        !python -m pip -q install ai-tools
        !python -m pip -q install cognitive-face
        progress_bar.update(2)
        !python -m pip -q install zprint
        !python -m pip -q install kornia==0.5.0
        import importlib
        progress_bar.update(2) #40
        !python -m pip -q install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
        !python -m pip -q install -e git+https://github.com/openai/CLIP.git@main#egg=clip
        progress_bar.update(2)
        !python -m pip -q install lpips
        !python -m pip -q install keras
        progress_bar.update(2) #50
        gitclone('https://github.com/Sxela/k-diffusion')
        os.chdir( f'./k-diffusion')
        subprocess.run(['git', 'pull'])
        !python -m pip -q install -e .
        os.chdir( f'../')
        import sys
        sys.path.append('./k-diffusion')
        progress_bar.update(1) #60
        !python -m pip -q install wget
        !python -m pip -q install webdataset
        progress_bar.update(2)
        !python -m pip -q install open_clip_torch
        !python -m pip -q install opencv-python==4.5.5.64
        progress_bar.update(2)
        !python -m pip -q uninstall torchtext -y
        !python -m pip -q install pandas matplotlib
        progress_bar.update(2)
        !python -m pip -q install fvcore








