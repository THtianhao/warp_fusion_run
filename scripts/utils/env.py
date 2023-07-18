import os
import sys

from scripts.utils.path import createPath

root_path = os.getcwd()
root_dir = os.getcwd()
sys.path.append(f'{root_dir}/python-color-transfer')
sys.path.append(f'{root_dir}/k-diffusion')
sys.path.append(f'{root_dir}/guided-diffusion')
sys.path.append(f'{root_dir}/ResizeRight')
sys.path.append(f'{root_dir}/BLIP')
sys.path.append(f'{root_dir}/ControlNet')

sys.path.append(root_path)
initDirPath = os.path.join(root_path, 'init_images')
createPath(initDirPath)
outDirPath = os.path.join(root_path, 'images_out')
createPath(outDirPath)
model_path = f'{root_path}/models'
createPath(model_path)
toto_dir = f'{root_dir}/toto_out'
createPath(toto_dir)

createPath('./embeddings')
