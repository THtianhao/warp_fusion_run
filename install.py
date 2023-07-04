import subprocess
import os
from utils.cmd import pipi, pipie, pipis, gitclone

from utils.nvidia_utils import print_card

if __name__ == "__main__":
    print_card()
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
    pipie('./stablediffusion')
