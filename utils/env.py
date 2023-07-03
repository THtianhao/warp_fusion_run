import os

from utils.path import createPath
root_path = os.getcwd()

initDirPath = os.path.join(root_path,'init_images')
createPath(initDirPath)
outDirPath = os.path.join(root_path,'images_out')
createPath(outDirPath)
root_dir = os.getcwd()
model_path = f'{root_path}/models'
createPath(model_path)

createPath('./embeddings')
