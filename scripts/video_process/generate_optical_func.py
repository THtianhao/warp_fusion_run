# @title Generate optical flow and consistency maps
# @markdown Run once per init video and width_height setting.
# if you're running locally, just restart this runtime, no need to edit PIL files.
import os
import pathlib
import subprocess
from glob import glob

import PIL
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from annotator.uniformer.mmcv import DataLoader
from scripts.bean.video_config_bean import VideoConfigBean
from scripts.settings.setting import side_x, side_y
from scripts.utils.env import root_dir
from scripts.video_process.color_transfor_func import flow_to_image, save_preview, make_cc_map, vstack, hstack, fit
from scripts.video_process.flow_datasets import flowDataset

flow_warp = True
check_consistency = True
force_flow_generation = False  # @param {type:'boolean'}

use_legacy_cc = False  # @param{'type':'boolean'}
threads = 4  # @param {'type':'number'}
# @markdown If you're having "process died" error on Windows, set num_workers to 0
num_workers = 0  # @param {'type':'number'}

# @markdown Use lower quality model (half-precision).\
# @markdown Uses half the vram, allows fitting 1500x1500+ frames into 16gigs, which the original full-precision RAFT can't do.
flow_lq = True  # @param {type:'boolean'}
# @markdown Save human-readable flow images along with motion vectors. Check /{your output dir}/videoFrames/out_flo_fwd folder.
flow_save_img_preview = False  # @param {type:'boolean'}

# #@markdown reverse_cc_order - on - default value (like in older notebooks). off - reverses consistency computation
reverse_cc_order = True  #
# #@param {type:'boolean'}
if not flow_warp: print('flow_wapr not set, skipping')
# @markdown Use previous pre-compile raft version (won't work with pytorch 2.0)
use_jit_raft = False  # @param {'type':'boolean'}
# @markdown Compile raft model (only with use_raft_jit = False). Compiles the model (~about 2 minutes) for ~30% speedup. Use for very long runs.
compile_raft = False  # @param {'type':'boolean'}
# @markdown Flow estimation quality (number of iterations, 12 - default. higher - better and slower)
num_flow_updates = 12  # @param {'type':'number'}
# \@markdown Unreliable areas mask (missed consistency) width
# \@markdown Default = 1
missed_consistency_dilation = 2  # \ @param {'type':'number'}
# \@markdown Motion edge areas (edge consistency) width
# \@markdown Default = 11
edge_consistency_width = 11  # \@param {'type':'number'}

from multiprocessing.pool import ThreadPool as Pool
import gc
def generate_optical_flow(bean:VideoConfigBean):
    # if (animation_mode == 'Video Input') and (flow_warp):
    def flow_batch(i, batch, pool):
        with torch.cuda.amp.autocast():
            batch = batch[0]
            frame_1 = batch[0][None, ...].cuda()
            frame_2 = batch[1][None, ...].cuda()
            frame1 = ds.frames[i]
            frame1 = frame1.replace('\\', '/')
            out_flow21_fn = f"{flo_fwd_folder}/{frame1.split('/')[-1]}"
            if flow_lq:   frame_1, frame_2 = frame_1, frame_2
            if use_jit_raft:
                _, flow21 = raft_model(frame_2, frame_1)
            else:
                flow21 = raft_model(frame_2, frame_1, num_flow_updates=num_flow_updates)[-1]  # flow_bwd
            flow21 = flow21[0].permute(1, 2, 0).detach().cpu().numpy()

            if flow_save_img_preview or i in range(0, len(ds), len(ds) // 10):
                pool.apply_async(save_preview, (flow21, out_flow21_fn + '.jpg'))
            pool.apply_async(np.save, (out_flow21_fn, flow21))
            if check_consistency:
                if use_jit_raft:
                    _, flow12 = raft_model(frame_1, frame_2)
                else:
                    flow12 = raft_model(frame_1, frame_2)[-1]  # flow_fwd

                flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()
                if flow_save_img_preview:
                    pool.apply_async(save_preview, (flow12, out_flow21_fn + '_12' + '.jpg'))
                if use_legacy_cc:
                    pool.apply_async(np.save, (out_flow21_fn + '_12', flow12))
                else:
                    joint_mask = make_cc_map(flow12, flow21, dilation=missed_consistency_dilation,
                                             edge_width=edge_consistency_width)
                    joint_mask = PIL.Image.fromarray(joint_mask.astype('uint8'))
                    cc_path = f"{flo_fwd_folder}/{frame1.split('/')[-1]}-21_cc.jpg"
                    # print(cc_path)
                    joint_mask.save(cc_path)
                    # pool.apply_async(joint_mask.save, cc_path)
    raft_model = None
    in_path = bean.videoFramesFolder if not bean.flow_video_init_path else bean.flowVideoFramesFolder
    flo_fwd_folder = flo_folder = in_path + f'_out_flo_fwd/{side_x}_{side_y}/'
    print(flo_folder)
    flows = glob(flo_folder + '/*.*')
    if (len(flows) > 0) and not force_flow_generation: print(
        f'Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {flo_folder}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again.')

    if (len(flows) == 0) or force_flow_generation:
        ds = flowDataset(in_path, normalize=not use_jit_raft)

        frames = sorted(glob(in_path + '/*.*'));
        if len(frames) < 2:
            print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')
        if len(frames) >= 2:
            if __name__ == '__main__':
                dl = DataLoader(ds, num_workers=num_workers)
                if use_jit_raft:
                    if flow_lq:
                        raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_half.jit').eval()
                    # raft_model = torch.nn.DataParallel(RAFT(args2))
                    else:
                        raft_model = torch.jit.load(f'{root_dir}/WarpFusion/raft/raft_fp32.jit').eval()
                    # raft_model.load_state_dict(torch.load(f'{root_path}/RAFT/models/raft-things.pth'))
                    # raft_model = raft_model.module.cuda().eval()
                else:
                    if raft_model is None or not compile_raft:
                        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
                        from torchvision.models.optical_flow import raft_large, raft_small

                        raft_weights = Raft_Large_Weights.C_T_SKHT_V1
                        raft_device = "cuda" if torch.cuda.is_available() else "cpu"

                        raft_model = raft_large(weights=raft_weights, progress=False).to(raft_device)
                        # raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(raft_device)
                        raft_model = raft_model.eval()
                        # if gpu != 'T4' and compile_raft: raft_model = torch.compile(raft_model)
                        # if flow_lq:
                        #     raft_model = raft_model.half()

                temp_flo = in_path + '_temp_flo'
                # flo_fwd_folder = in_path+'_out_flo_fwd'
                flo_fwd_folder = in_path + f'_out_flo_fwd/{side_x}_{side_y}/'
                for f in pathlib.Path(f'{flo_fwd_folder}').glob('*.*'):
                    f.unlink()

                os.makedirs(flo_fwd_folder, exist_ok=True)
                os.makedirs(temp_flo, exist_ok=True)
                cc_path = f'{root_dir}/flow_tools/check_consistency.py'
                with torch.no_grad():
                    p = Pool(threads)
                    for i, batch in enumerate(tqdm(dl)):
                        flow_batch(i, batch, p)
                    p.close()
                    p.join()

                del raft_model, p, dl, ds
                gc.collect()
                if check_consistency and use_legacy_cc:
                    fwd = f"{flo_fwd_folder}/*jpg.npy"
                    bwd = f"{flo_fwd_folder}/*jpg_12.npy"

                    if reverse_cc_order:
                        # old version, may be incorrect
                        print('Doing bwd->fwd cc check')
                        cmd = f'python {cc_path} --flow_fwd {fwd} --flow_bwd {bwd} --output {flo_fwd_folder} --image_output --output_postfix=-"21_cc" --blur = 0. --save_separate_channels --skip_numpy_output'
                        subprocess.run(cmd)
                    else:
                        print('Doing fwd->bwd cc check')
                        cmd = f'python {cc_path} --flow_fwd {bwd} --flow_bwd {fwd} --output {flo_fwd_folder} --image_output --output_postfix=-"21_cc" --blur = 0. --save_separate_channels --skip_numpy_output'
                        subprocess.run(cmd)
                    # delete forward flow
                    # for f in pathlib.Path(flo_fwd_folder).glob('*jpg_12.npy'):
                    #   f.unlink()
    flo_imgs = glob(flo_fwd_folder + '/*.jpg.jpg')[:5]
    vframes = []
    for flo_img in flo_imgs:
        hframes = []
        flo_img = flo_img.replace('\\', '/')
        frame = Image.open(bean.videoFramesFolder + '/' + flo_img.split('/')[-1][:-4])
        hframes.append(frame)
        try:
            alpha = Image.open(bean.videoFramesAlpha + '/' + flo_img.split('/')[-1][:-4]).resize(frame.size)
            hframes.append(alpha)
        except:
            pass
        try:
            cc_img = Image.open(flo_img[:-4] + '-21_cc.jpg').convert('L').resize(frame.size)
            hframes.append(cc_img)
        except:
            pass
        try:
            flo_img = Image.open(flo_img).resize(frame.size)
            hframes.append(flo_img)
        except:
            pass
        v_imgs = vstack(hframes)
        vframes.append(v_imgs)
    preview = hstack(vframes)
    del vframes, hframes
    fit(preview, 1024)