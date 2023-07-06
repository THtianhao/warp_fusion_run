# @title Setup Optical Flow
##@markdown Run once per session. Doesn't download again if model path exists.
##@markdown Use force download to reload raft models if needed
import math
import os

from scripts.sd_model_select import sd_model
from scripts.settings.main_settings import warp_mode, match_color_strength, consistency_blur, consistency_dilate, use_patchmatch_inpaiting, missed_consistency_weight, overshoot_consistency_weight, \
    edges_consistency_weight
from scripts.settings.setting import warp_interp
from scripts.utils.env import root_dir, root_path
from scripts.video_process.Input_padder import InputPadder

force_download = False  # \@param {type:'boolean'}
# import wget
import zipfile, shutil

# @title Define color matching and brightness adjustment
os.chdir(f"{root_dir}/python-color-transfer")
from python_color_transfer.color_transfer import ColorTransfer, Regrain

os.chdir(root_path)

PT = ColorTransfer()
RG = Regrain()
# @markdown ###Automatic Brightness Adjustment
# @markdown Automatically adjust image brightness when its mean value reaches a certain threshold\
# @markdown Ratio means the vaue by which pixel values are multiplied when the thresjold is reached\
# @markdown Fix amount is being directly added to\subtracted from pixel values to prevent oversaturation due to multiplications\
# @markdown Fix amount is also being applied to border values defined by min\max threshold, like 1 and 254 to keep the image from having burnt out\pitch black areas while still being within set high\low thresholds


# @markdown The idea comes from https://github.com/lowfuel/progrockdiffusion

enable_adjust_brightness = False  # @param {'type':'boolean'}
high_brightness_threshold = 180  # @param {'type':'number'}
high_brightness_adjust_ratio = 0.97  # @param {'type':'number'}
high_brightness_adjust_fix_amount = 2  # @param {'type':'number'}
max_brightness_threshold = 254  # @param {'type':'number'}
low_brightness_threshold = 40  # @param {'type':'number'}
low_brightness_adjust_ratio = 1.03  # @param {'type':'number'}
low_brightness_adjust_fix_amount = 2  # @param {'type':'number'}
min_brightness_threshold = 1  # @param {'type':'number'}

def match_color(stylized_img, raw_img, opacity=1.):
    if opacity > 0:
        img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
        img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
        # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
        img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        img_arr_reg = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
        img_arr_reg = img_arr_reg * opacity + img_arr_in * (1 - opacity)
        img_arr_reg = cv2.cvtColor(img_arr_reg.round().astype('uint8'), cv2.COLOR_BGR2RGB)
        return img_arr_reg
    else:
        return raw_img

from PIL import Image, ImageStat, ImageEnhance, ImageDraw

def get_stats(image):
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / len(stat.mean)
    contrast = sum(stat.stddev) / len(stat.stddev)
    return brightness, contrast

# implemetation taken from https://github.com/lowfuel/progrockdiffusion

def adjust_brightness(image):
    brightness, contrast = get_stats(image)
    if brightness > high_brightness_threshold:
        print(" Brightness over threshold. Compensating!")
        filter = ImageEnhance.Brightness(image)
        image = filter.enhance(high_brightness_adjust_ratio)
        image = np.array(image)
        image = np.where(image > high_brightness_threshold, image - high_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
        image = Image.fromarray(image)
    if brightness < low_brightness_threshold:
        print(" Brightness below threshold. Compensating!")
        filter = ImageEnhance.Brightness(image)
        image = filter.enhance(low_brightness_adjust_ratio)
        image = np.array(image)
        image = np.where(image < low_brightness_threshold, image + low_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
        image = Image.fromarray(image)

    image = np.array(image)
    image = np.where(image > max_brightness_threshold, image - high_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
    image = np.where(image < min_brightness_threshold, image + low_brightness_adjust_fix_amount, image).clip(0, 255).round().astype('uint8')
    image = Image.fromarray(image)
    return image

# @title Define optical flow functions for Video input animation mode only
# if animation_mode == 'Video Input Legacy':
DEBUG = False

# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k0 = np.clip(k0, 0, colorwheel.shape[0] - 1)
    k1 = k0 + 1
    k1 = np.clip(k1, 0, colorwheel.shape[0] - 1)
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

from torch import Tensor

import cv2

def extract_occlusion_mask(flow, threshold=10):
    flow = flow.clone()[0].permute(1, 2, 0).detach().cpu().numpy()
    h, w = flow.shape[:2]

    """
    Extract a mask containing all the points that have no origin in frame one.

    Parameters:
        motion_vector (numpy.ndarray): A 2D array of motion vectors.
        threshold (int): The threshold value for the magnitude of the motion vector.

    Returns:
        numpy.ndarray: The occlusion mask.
    """
    # Compute the magnitude of the motion vector.
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to identify occlusions.
    occlusion_mask = (mag > threshold).astype(np.uint8)

    return occlusion_mask, mag

import cv2
import numpy as np

def edge_detector(image, threshold=0.5, edge_width=1):
    """
    Detect edges in an image with adjustable edge width.

    Parameters:
        image (numpy.ndarray): The input image.
        edge_width (int): The width of the edges to detect.

    Returns:
        numpy.ndarray: The edge image.
    """
    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Sobel edge map.
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=edge_width)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=edge_width)

    # Compute the edge magnitude.
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Normalize the magnitude to the range [0, 1].
    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold the magnitude to create a binary edge image.

    edge_image = (mag > threshold).astype(np.uint8) * 255

    return edge_image

def get_unreliable(flow):
    # Mask pixels that have no source and will be taken from frame1, to remove trails and ghosting.

    # flow = flow[0].cpu().numpy().transpose(1,2,0)

    # Calculate the coordinates of pixels in the new frame
    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = x + flow[..., 0]
    new_y = y + flow[..., 1]

    # Create a mask for the valid pixels in the new frame
    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)

    # Create the new frame by interpolating the pixel values using the calculated coordinates
    new_frame = np.zeros((flow.shape[0], flow.shape[1], 3)) * 1. - 1
    new_frame[new_y[mask].astype(np.int32), new_x[mask].astype(np.int32)] = 255

    # Keep masked area, discard the image.
    new_frame = new_frame == -1
    return new_frame, mask

from scipy.ndimage import binary_fill_holes
from skimage.morphology import disk, binary_erosion, binary_dilation, binary_opening, binary_closing

import cv2

def remove_small_holes(mask, min_size=50):
    # Copy the input binary mask
    result = mask.copy()
    # Find contours of connected components in the binary image
    contours, hierarchy = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over each contour
    for i in range(len(contours)):
        # Compute the area of the i-th contour
        area = cv2.contourArea(contours[i])

        # Check if the area of the i-th contour is smaller than min_size
        if area < min_size:
            # Draw a filled contour over the i-th contour region
            cv2.drawContours(result, [contours[i]], 0, 255, -1, cv2.LINE_AA, hierarchy, 0)

    return result

def filter_unreliable(mask, dilation=1):
    img = 255 - remove_small_holes((1 - mask[..., 0].astype('uint8')) * 255, 200)
    # img = binary_fill_holes(img)
    img = binary_erosion(img, disk(1))
    img = binary_dilation(img, disk(dilation))
    return img

def make_cc_map(predicted_flows, predicted_flows_bwd, dilation=1, edge_width=11):
    flow_imgs = flow_to_image(predicted_flows_bwd)
    edge = edge_detector(flow_imgs.astype('uint8'), threshold=0.1, edge_width=edge_width)
    res, _ = get_unreliable(predicted_flows)
    _, overshoot = get_unreliable(predicted_flows_bwd)
    joint_mask = np.ones_like(res) * 255
    joint_mask[..., 0] = 255 - (filter_unreliable(res, dilation) * 255)
    joint_mask[..., 1] = (overshoot * 255)
    joint_mask[..., 2] = 255 - edge

    return joint_mask

import numpy as np
import argparse, PIL, cv2
from PIL import Image
import torch
import scipy.ndimage

args2 = argparse.Namespace()
args2.small = False
args2.mixed_precision = True

TAG_CHAR = np.array([202021.25], np.float32)

def writeFlow(filename, uv, v=None):
    """
    https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
    Copyright 2017 NVIDIA CORPORATION

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def load_cc(path,controlnetDebugFolder, args, blur=2, dilate=0, ):
    multilayer_weights = np.array(Image.open(path)) / 255
    weights = np.ones_like(multilayer_weights[..., 0])
    weights *= multilayer_weights[..., 0].clip(1 - missed_consistency_weight, 1)
    weights *= multilayer_weights[..., 1].clip(1 - overshoot_consistency_weight, 1)
    weights *= multilayer_weights[..., 2].clip(1 - edges_consistency_weight, 1)
    weights = np.where(weights < 0.5, 0, 1)
    if dilate > 0:
        weights = (1 - binary_dilation(1 - weights, disk(dilate))).astype('uint8')
    if blur > 0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    weights = np.repeat(weights[..., None], 3, axis=2)
    # print('------------cc debug------', f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_cc_mask.jpg')
    PIL.Image.fromarray((weights * 255).astype('uint8')).save(f'{controlnetDebugFolder}/{args.batch_name}({args.batchNum})_cc_mask.jpg', quality=95)
    # assert False
    if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
    return weights

def load_img(img, size):
    img = Image.open(img).convert('RGB').resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()

def get_flow(frame1, frame2, model, iters=20, half=True):
    # print(frame1.shape, frame2.shape)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    if half: frame1, frame2 = frame1, frame2
    # print(frame1.shape, frame2.shape)
    _, flow12 = model(frame1, frame2)
    flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

    return flow12

def warp_flow(img, flow, mul=1.):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    flow *= mul
    res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)

    return res

def makeEven(_x):
    return _x if (_x % 2 == 0) else _x + 1

def fit(img, maxsize=512):
    maxdim = max(*img.size)
    if maxdim > maxsize:
        # if True:
        ratio = maxsize / maxdim
        x, y = img.size
        size = (makeEven(int(x * ratio)), makeEven(int(y * ratio)))
        img = img.resize(size, warp_interp)
    return img

def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0.,
         pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    if isinstance(flo_path, str):
        flow21 = np.load(flo_path)
    else:
        flow21 = flo_path
    # print('loaded flow from ', flo_path, ' witch shape ', flow21.shape)
    pad = int(max(flow21.shape) * pad_pct)
    flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant')
    # print('frame1.size, frame2.size, padded flow21.shape')
    # print(frame1.size, frame2.size, flow21.shape)

    frame1pil = np.array(frame1.convert('RGB'))  # .resize((flow21.shape[1]-pad*2,flow21.shape[0]-pad*2),warp_interp))
    frame1pil = np.pad(frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode)
    if video_mode:
        warp_mul = 1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0] - pad, pad:frame1_warped21.shape[1] - pad, :]

    frame2pil = np.array(frame2.convert('RGB').resize((flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp))
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path:
        forward_weights = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)
        # print('forward_weights')
        # print(forward_weights.shape)
        if not video_mode and match_color_strength > 0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)

        forward_weights = forward_weights.clip(forward_clip, 1.)
        if use_patchmatch_inpaiting > 0 and warp_mode == 'use_image':
            print('PatchMatch disabled.')
            # if not video_mode and is_colab:
            #       print('patchmatching')
            #       # print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
            #       patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
            #       frame2pil = np.array(frame2pil)*(1-use_patchmatch_inpaiting)+use_patchmatch_inpaiting*np.array(patch_match.inpaint(frame1_warped21, patchmatch_mask, patch_size=5))
            #       # blended_w = Image.fromarray(blended_w)
        blended_w = frame2pil * (1 - blend) + blend * (frame1_warped21 * forward_weights + frame2pil * (1 - forward_weights))
    else:
        if not video_mode and match_color_strength > 0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)

    blended_w = Image.fromarray(blended_w.round().astype('uint8'))
    # if use_patchmatch_inpaiting and warp_mode == 'use_image':
    #           print('patchmatching')
    #           print(np.array(blended_w).shape, forward_weights[...,0][...,None].shape )
    #           patchmatch_mask = (forward_weights[...,0][...,None]*-255.+255).astype('uint8')
    #           blended_w = patch_match.inpaint(blended_w, patchmatch_mask, patch_size=5)
    #           blended_w = Image.fromarray(blended_w)
    if not video_mode:
        if enable_adjust_brightness: blended_w = adjust_brightness(blended_w)
    return blended_w

def warp_lat(frame1, frame2, flo_path, blend=0.5, weights_path=None, forward_clip=0.,
             pad_pct=0.1, padding_mode='reflect', inpaint_blend=0., video_mode=False, warp_mul=1.):
    warp_downscaled = True
    flow21 = np.load(flo_path)
    pad = int(max(flow21.shape) * pad_pct)
    if warp_downscaled:
        flow21 = flow21.transpose(2, 0, 1)[None, ...]
        flow21 = torch.nn.functional.interpolate(torch.from_numpy(flow21).float(), scale_factor=1 / 8, mode='bilinear')
        flow21 = flow21.numpy()[0].transpose(1, 2, 0) / 8
        # flow21 = flow21[::8,::8,:]/8

    flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant')

    if not warp_downscaled:
        frame1 = torch.nn.functional.interpolate(frame1, scale_factor=8)
    frame1pil = frame1.cpu().numpy()[0].transpose(1, 2, 0)

    frame1pil = np.pad(frame1pil, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padding_mode)
    if video_mode:
        warp_mul = 1.
    frame1_warped21 = warp_flow(frame1pil, flow21, warp_mul)
    frame1_warped21 = frame1_warped21[pad:frame1_warped21.shape[0] - pad, pad:frame1_warped21.shape[1] - pad, :]
    if not warp_downscaled:
        frame2pil = frame2.convert('RGB').resize((flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp)
    else:
        frame2pil = frame2.convert('RGB').resize(((flow21.shape[1] - pad * 2) * 8, (flow21.shape[0] - pad * 2) * 8), warp_interp)
    frame2pil = np.array(frame2pil)
    frame2pil = (frame2pil / 255.)[None, ...].transpose(0, 3, 1, 2)
    frame2pil = 2 * torch.from_numpy(frame2pil).float().cuda() - 1.
    frame2pil = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(frame2pil))
    if not warp_downscaled: frame2pil = torch.nn.functional.interpolate(frame2pil, scale_factor=8)
    frame2pil = frame2pil.cpu().numpy()[0].transpose(1, 2, 0)
    # if not video_mode: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
    if weights_path:
        forward_weights = load_cc(weights_path, blur=consistency_blur, dilate=consistency_dilate)
        print(forward_weights[..., :1].shape, 'forward_weights.shape')
        forward_weights = np.repeat(forward_weights[..., :1], 4, axis=-1)
        # print('forward_weights')
        # print(forward_weights.shape)
        print('frame2pil.shape, frame1_warped21.shape, flow21.shape', frame2pil.shape, frame1_warped21.shape, flow21.shape)
        forward_weights = forward_weights.clip(forward_clip, 1.)
        if warp_downscaled: forward_weights = forward_weights[::8, ::8, :]; print(forward_weights.shape, 'forward_weights.shape')
        blended_w = frame2pil * (1 - blend) + blend * (frame1_warped21 * forward_weights + frame2pil * (1 - forward_weights))
    else:
        if not video_mode and not warp_mode == 'use_latent' and match_color_strength > 0.: frame2pil = match_color(frame1_warped21, frame2pil, opacity=match_color_strength)
        blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)
    blended_w = blended_w.transpose(2, 0, 1)[None, ...]
    blended_w = torch.from_numpy(blended_w).float()
    if not warp_downscaled:
        # blended_w = blended_w[::8,::8,:]
        blended_w = torch.nn.functional.interpolate(blended_w, scale_factor=1 / 8, mode='bilinear')
    return blended_w  # torch.nn.functional.interpolate(torch.from_numpy(blended_w), scale_factor = 1/8)
os.chdir(root_path)
















def hstack(images):
    if isinstance(images[0], str):
        images = [Image.open(image).convert('RGB') for image in images]
    widths, heights = zip(*(i.size for i in images))
    for image in images:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((0, 00), (image.size[0], image.size[1])), outline="black", width=3)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

import locale

def getpreferredencoding(do_setlocale=True):
    return "UTF-8"

def vstack(images):
    if isinstance(next(iter(images)), str):
        images = [Image.open(image).convert('RGB') for image in images]
    widths, heights = zip(*(i.size for i in images))

    total_height = sum(heights)
    max_width = max(widths)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im

from torch.utils.data import DataLoader

def save_preview(flow21, out_flow21_fn):
    try:
        Image.fromarray(flow_to_image(flow21)).save(out_flow21_fn, quality=90)
    except:
        print('Error saving flow preview for frame ', out_flow21_fn)

# copyright Alex Spirin @ 2022
def blended_roll(img_copy, shift, axis):
    if int(shift) == shift:
        return np.roll(img_copy, int(shift), axis=axis)

    max = math.ceil(shift)
    min = math.floor(shift)
    if min != 0:
        img_min = np.roll(img_copy, min, axis=axis)
    else:
        img_min = img_copy
    img_max = np.roll(img_copy, max, axis=axis)
    blend = max - shift
    img_blend = img_min * blend + img_max * (1 - blend)
    return img_blend

# copyright Alex Spirin @ 2022
def move_cluster(img, i, res2, center, mode='blended_roll'):
    img_copy = img.copy()
    motion = center[i]
    mask = np.where(res2 == motion, 1, 0)[..., 0][..., None]
    y, x = motion
    if mode == 'blended_roll':
        img_copy = blended_roll(img_copy, x, 0)
        img_copy = blended_roll(img_copy, y, 1)
    if mode == 'int_roll':
        img_copy = np.roll(img_copy, int(x), axis=0)
        img_copy = np.roll(img_copy, int(y), axis=1)
    return img_copy, mask

import cv2

def get_k(flow, K):
    Z = flow.reshape((-1, 2))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    res = center[label.flatten()]
    res2 = res.reshape((flow.shape))
    return res2, center

def k_means_warp(flo, img, num_k):
    # flo = np.load(flo)
    img = np.array((img).convert('RGB'))
    num_k = 8

    # print(img.shape)
    res2, center = get_k(flo, num_k)
    center = sorted(list(center), key=lambda x: abs(x).mean())

    img = cv2.resize(img, (res2.shape[:-1][::-1]))
    img_out = np.ones_like(img) * 255.

    for i in range(num_k):
        img_rolled, mask_i = move_cluster(img, i, res2, center)
        img_out = img_out * (1 - mask_i) + img_rolled * (mask_i)

    # cv2_imshow(img_out)
    return Image.fromarray(img_out.astype('uint8'))
