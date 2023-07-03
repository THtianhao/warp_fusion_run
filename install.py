import subprocess
from utils.cmd import pipis

from utils.nvidia_utils import print_card



if __name__ == "__main__":
    print_card()
    pipis(['tqdm', 'ipywidgets==7.7.1','protobuf==3.20.3'])







