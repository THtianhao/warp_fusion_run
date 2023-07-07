from typing import Mapping

import mediapipe as mp

import numpy
import torch
import cv2
import io
import math
from PIL import ImageOps
import requests
import json
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from resize_right import resize
import numpy as np
from scripts.settings.setting import side_x, side_y, skip_augs
from scripts.settings.no_gui_config import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
device = DEVICE  # At least one of the modules expects this name..


def append_dims(x, n):
    return x[(Ellipsis, *(None, ) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection  # Only for counting faces.
mp_face_mesh = mp.solutions.face_mesh
mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

f_thick = 2
f_rad = 1
right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)
# mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
face_connection_spec = {}
for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
    face_connection_spec[edge] = head_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
    face_connection_spec[edge] = left_eye_draw
for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
    face_connection_spec[edge] = left_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
#    face_connection_spec[edge] = left_iris_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
    face_connection_spec[edge] = right_eye_draw
for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
    face_connection_spec[edge] = right_eyebrow_draw
# for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
#    face_connection_spec[edge] = right_iris_draw
for edge in mp_face_mesh.FACEMESH_LIPS:
    face_connection_spec[edge] = mouth_draw
iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < 0.9) or (landmark.HasField('presence') and landmark.presence < 0.5)):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols * landmark.x)
        image_y = int(image_rows * landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y - halfwidth:image_y + halfwidth, image_x - halfwidth:image_x + halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(input_image: Image.Image, max_faces: int, min_face_size_pixels: int = 0, return_annotation_data: bool = False):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    If return_annotation_data is TRUE (default: false) then in addition to returning the 'detected face' image, three
    additional parameters will be returned: faces before filtering, faces after filtering, and an annotation image.
    The faces_before_filtering return value is the number of faces detected in an image with no filtering.
    faces_after_filtering is the number of faces remaining after filtering small faces.
    :return:
      If 'return_annotation_data==True', returns (numpy array, numpy array, int, int).
      If 'return_annotation_data==False' (default), returns a numpy array.
    """
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
    ) as facemesh:
        img_rgb = numpy.asarray(input_image)
        results = facemesh.process(img_rgb).multi_face_landmarks
        if results is None:
            return None
        faces_found_before_filtering = len(results)

        # Filter faces that are too small
        filtered_landmarks = []
        for lm in results:
            landmarks = lm.landmark
            face_rect = [
                landmarks[0].x,
                landmarks[0].y,
                landmarks[0].x,
                landmarks[0].y,
            ]  # Left, up, right, down.
            for i in range(len(landmarks)):
                face_rect[0] = min(face_rect[0], landmarks[i].x)
                face_rect[1] = min(face_rect[1], landmarks[i].y)
                face_rect[2] = max(face_rect[2], landmarks[i].x)
                face_rect[3] = max(face_rect[3], landmarks[i].y)
            if min_face_size_pixels > 0:
                face_width = abs(face_rect[2] - face_rect[0])
                face_height = abs(face_rect[3] - face_rect[1])
                face_width_pixels = face_width * input_image.size[0]
                face_height_pixels = face_height * input_image.size[1]
                face_size = min(face_width_pixels, face_height_pixels)
                if face_size >= min_face_size_pixels:
                    filtered_landmarks.append(lm)
            else:
                filtered_landmarks.append(lm)

        faces_remaining_after_filtering = len(filtered_landmarks)

        # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
        empty = numpy.zeros_like(img_rgb)

        # Draw detected faces:
        for face_landmarks in filtered_landmarks:
            mp_drawing.draw_landmarks(empty, face_landmarks, connections=face_connection_spec.keys(), landmark_drawing_spec=None, connection_drawing_spec=face_connection_spec)
            draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

        # Flip BGR back to RGB.
        empty = reverse_channels(empty)

        # We might have to generate a composite.
        annotated = None
        if return_annotation_data:
            # Note that we're copying the input image AND flipping the channels so we can draw on top of it.
            annotated = reverse_channels(numpy.asarray(input_image)).copy()
            for face_landmarks in filtered_landmarks:
                mp_drawing.draw_landmarks(empty, face_landmarks, connections=face_connection_spec.keys(), landmark_drawing_spec=None, connection_drawing_spec=face_connection_spec)
                draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)
            annotated = reverse_channels(annotated)

        if not return_annotation_data:
            return empty
        else:
            return empty, annotated, faces_found_before_filtering, faces_remaining_after_filtering


# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2**len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(prelin_mode, batch_size):
    if prelin_mode == 'color':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False)
    elif prelin_mode == 'gray':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):

    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


cutout_debug = False
padargs = {}


class MakeCutoutsDango(nn.Module):

    def __init__(self, args, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        if args.animation_mode == 'None':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif args.animation_mode == 'Video Input Legacy':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif args.animation_mode == '2D' or args.animation_mode == 'Video Input':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.4),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
            ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape_2 = [1, 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(input, ((sideY - max_size) // 2, (sideY - max_size) // 2, (sideX - max_size) // 2, (sideX - max_size) // 2), **padargs)
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save("cutout_overview0.jpg", quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save("cutout_InnerCrop.jpg", quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True: cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


stop_on_next_loop = False  # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
TRANSLATION_SCALE = 1.0 / 200.0


def get_sched_from_json(frame_num, sched_json, blend=False):
    frame_num = int(frame_num)
    frame_num = max(frame_num, 0)
    sched_int = {}
    for key in sched_json.keys():
        sched_int[int(key)] = sched_json[key]
    sched_json = sched_int
    keys = sorted(list(sched_json.keys()))
    # print(keys)
    if frame_num < 0:
        frame_num = max(keys)
    try:
        frame_num = min(frame_num, max(keys))  # clamp frame num to 0:max(keys) range
    except:
        pass

    # print('clamped frame num ', frame_num)
    if frame_num in keys:
        return sched_json[frame_num]
        # print('frame in keys')
    if frame_num not in keys:
        for i in range(len(keys) - 1):
            k1 = keys[i]
            k2 = keys[i + 1]
            if frame_num > k1 and frame_num < k2:
                if not blend:
                    print('frame between keys, no blend')
                    return sched_json[k1]
                if blend:
                    total_dist = k2 - k1
                    dist_from_k1 = frame_num - k1
                    return sched_json[k1] * (1 - dist_from_k1 / total_dist) + sched_json[k2] * (dist_from_k1 / total_dist)
            # else: print(f'frame {frame_num} not in {k1} {k2}')
    return 0


def get_scheduled_arg(frame_num, schedule):
    if isinstance(schedule, list):
        return schedule[frame_num] if frame_num < len(schedule) else schedule[-1]
    if isinstance(schedule, dict):
        return get_sched_from_json(frame_num, schedule, blend=blend_json_schedules)


import piexif


def json2exif(settings):
    settings = json.dumps(settings)
    exif_ifd = {piexif.ExifIFD.UserComment: settings.encode()}
    exif_dict = {"Exif": exif_ifd}
    exif_dat = piexif.dump(exif_dict)
    return exif_dat


def img2tensor(img, size=None):
    img = img.convert('RGB')
    if size: img = img.resize(size, warp_interp)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def warp_towards_init_fn(sample_pil, init_image):
    print('sample, init', type(sample_pil), type(init_image))
    size = sample_pil.size
    sample = img2tensor(sample_pil)
    init_image = img2tensor(init_image, size)
    flo = get_flow(init_image, sample, raft_model, half=flow_lq)
    # flo = get_flow(sample, init_image, raft_model, half=flow_lq)
    warped = warp(sample_pil,
                  sample_pil,
                  flo_path=flo,
                  blend=1,
                  weights_path=None,
                  forward_clip=0,
                  pad_pct=padding_ratio,
                  padding_mode=padding_mode,
                  inpaint_blend=inpaint_blend,
                  warp_mul=warp_strength)
    return warped


def do_3d_step(img_filepath, frame_num, forward_clip):
    global warp_mode, filename, match_frame, first_frame
    global first_frame_source
    if warp_mode == 'use_image':
        prev = Image.open(img_filepath)
    # if warp_mode == 'use_latent':
    #   prev = torch.load(img_filepath[:-4]+'_lat.pt')

    frame1_path = f'{videoFramesFolder}/{frame_num:06}.jpg'
    frame2 = Image.open(f'{videoFramesFolder}/{frame_num + 1:06}.jpg')

    flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}.npy"

    if flow_override_map not in [[], '', None]:
        mapped_frame_num = int(get_scheduled_arg(frame_num, flow_override_map))
        frame_override_path = f'{videoFramesFolder}/{mapped_frame_num:06}.jpg'
        flo_path = f"{flo_folder}/{frame_override_path.split('/')[-1]}.npy"

    if use_background_mask and not apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        # else:
        if VERBOSE: print('creating bg mask for frame ', frame_num)
        frame2 = apply_mask(frame2, frame_num, background, background_source, invert_mask)
        # frame2.save(f'frame2_{frame_num}.jpg')
    # init_image = 'warped.png'
    flow_blend = get_scheduled_arg(frame_num, flow_blend_schedule)
    printf('flow_blend: ', flow_blend, 'frame_num:', frame_num, 'len(flow_blend_schedule):', len(flow_blend_schedule))
    weights_path = None
    forward_clip = forward_weights_clip
    if check_consistency:
        if reverse_cc_order:
            weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
        else:
            weights_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

    if turbo_mode & (frame_num % int(turbo_steps) != 0):
        if forward_weights_clip_turbo_step:
            forward_clip = forward_weights_clip_turbo_step
        if disable_cc_for_turbo_frames:
            if VERBOSE: print('disabling cc for turbo frames')
            weights_path = None
    if warp_mode == 'use_image':
        prev = Image.open(img_filepath)

        if not warp_forward:
            printf('warping')
            warped = warp(prev,
                          frame2,
                          flo_path,
                          blend=flow_blend,
                          weights_path=weights_path,
                          forward_clip=forward_clip,
                          pad_pct=padding_ratio,
                          padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend,
                          warp_mul=warp_strength)
        else:
            flo_path = f"{flo_folder}/{frame1_path.split('/')[-1]}_12.npy"
            flo = np.load(flo_path)
            warped = k_means_warp(flo, prev, warp_num_k)
        if colormatch_frame != 'off' and not colormatch_after:
            if not turbo_mode & (frame_num % int(turbo_steps) != 0) or colormatch_turbo:
                try:
                    print('Matching color before warp to:')
                    filename = get_frame_from_color_mode(colormatch_frame, colormatch_offset, frame_num)
                    match_frame = Image.open(filename)
                    first_frame = match_frame
                    first_frame_source = filename

                except:
                    print(traceback.format_exc())
                    print(f'Frame with offset/position {colormatch_offset} not found')
                    if 'init' in colormatch_frame:
                        try:
                            filename = f'{videoFramesFolder}/{1:06}.jpg'
                            match_frame = Image.open(filename)
                            first_frame = match_frame
                            first_frame_source = filename
                        except:
                            pass
                print(f'Color matching the 1st frame before warp.')
                print('Colormatch source - ', first_frame_source)
                warped = Image.fromarray(match_color_var(first_frame, warped, opacity=color_match_frame_str, f=colormatch_method_fn, regrain=colormatch_regrain))
    if warp_mode == 'use_latent':
        prev = torch.load(img_filepath[:-4] + '_lat.pt')
        warped = warp_lat(prev,
                          frame2,
                          flo_path,
                          blend=flow_blend,
                          weights_path=weights_path,
                          forward_clip=forward_clip,
                          pad_pct=padding_ratio,
                          padding_mode=padding_mode,
                          inpaint_blend=inpaint_blend,
                          warp_mul=warp_strength)
    # warped = warped.resize((side_x,side_y), warp_interp)

    if use_background_mask and apply_mask_after_warp:
        # if turbo_mode & (frame_num % int(turbo_steps) != 0):
        #   print('disabling mask for turbo step, will be applied during turbo blend')
        #   return warped
        if VERBOSE: print('creating bg mask for frame ', frame_num)
        if warp_mode == 'use_latent':
            warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
        else:
            warped = apply_mask(warped, frame_num, background, background_source, invert_mask, warp_mode)
        # warped.save(f'warped_{frame_num}.jpg')

    return warped

def get_frame_from_color_mode(mode, offset, frame_num):
    if mode == 'color_video':
        if VERBOSE: print(f'the color video frame number {offset}.')
        filename = f'{colorVideoFramesFolder}/{offset + 1:06}.jpg'
    if mode == 'color_video_offset':
        if VERBOSE: print(f'the color video frame with offset {offset}.')
        filename = f'{colorVideoFramesFolder}/{frame_num - offset + 1:06}.jpg'
    if mode == 'stylized_frame_offset':
        if VERBOSE: print(f'the stylized frame with offset {offset}.')
        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{frame_num - offset:06}.png'
    if mode == 'stylized_frame':
        if VERBOSE: print(f'the stylized frame number {offset}.')
        filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{offset:06}.png'
        if not os.path.exists(filename):
            filename = f'{batchFolder}/{args.batch_name}({args.batchNum})_{args.start_frame + offset:06}.png'
    if mode == 'init_frame_offset':
        if VERBOSE: print(f'the raw init frame with offset {offset}.')
        filename = f'{videoFramesFolder}/{frame_num - offset + 1:06}.jpg'
    if mode == 'init_frame':
        if VERBOSE: print(f'the raw init frame number {offset}.')
        filename = f'{videoFramesFolder}/{offset + 1:06}.jpg'
    return filename


def apply_mask(
    init_image,
    frame_num,
    background,
    background_source,
    invert_mask=False,
    warp_mode='use_image',
):
    global mask_clip_low, mask_clip_high
    if warp_mode == 'use_image':
        size = init_image.size
    if warp_mode == 'use_latent':
        print(init_image.shape)
        size = init_image.shape[-1], init_image.shape[-2]
        size = [o * 8 for o in size]
        print('size', size)
    init_image_alpha = Image.open(f'{videoFramesAlpha}/{frame_num + 1:06}.jpg').resize(size).convert('L')
    if invert_mask:
        init_image_alpha = ImageOps.invert(init_image_alpha)
    if mask_clip_high < 255 or mask_clip_low > 0:
        arr = np.array(init_image_alpha)
        if mask_clip_high < 255:
            arr = np.where(arr < mask_clip_high, arr, 255)
        if mask_clip_low > 0:
            arr = np.where(arr > mask_clip_low, arr, 0)
        init_image_alpha = Image.fromarray(arr)

    if background == 'color':
        bg = Image.new('RGB', size, background_source)
    if background == 'image':
        bg = Image.open(background_source).convert('RGB').resize(size)
    if background == 'init_video':
        bg = Image.open(f'{videoFramesFolder}/{frame_num + 1:06}.jpg').resize(size)
    # init_image.putalpha(init_image_alpha)
    if warp_mode == 'use_image':
        bg.paste(init_image, (0, 0), init_image_alpha)
    if warp_mode == 'use_latent':
        # convert bg to latent

        bg = np.array(bg)
        bg = (bg / 255.)[None, ...].transpose(0, 3, 1, 2)
        bg = 2 * torch.from_numpy(bg).float().cuda() - 1.
        bg = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(bg))
        bg = bg.cpu().numpy()  # [0].transpose(1,2,0)
        init_image_alpha = np.array(init_image_alpha)[::8, ::8][None, None, ...]
        init_image_alpha = np.repeat(init_image_alpha, 4, axis=1) / 255
        print(bg.shape, init_image.shape, init_image_alpha.shape, init_image_alpha.max(), init_image_alpha.min())
        bg = init_image * init_image_alpha + bg * (1 - init_image_alpha)
    return bg


def softcap(arr, thresh=0.8, q=0.95):
    cap = torch.quantile(abs(arr).float(), q)
    printf('q -----', torch.quantile(abs(arr).float(), torch.Tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).cuda()))
    cap_ratio = (1 - thresh) / (cap - thresh)
    arr = torch.where(arr > thresh, thresh + (arr - thresh) * cap_ratio, arr)
    arr = torch.where(arr < -thresh, -thresh + (arr + thresh) * cap_ratio, arr)
    return arr



def save_settings(skip_save=False):
    settings_out = batchFolder + f"/settings"
    os.makedirs(settings_out, exist_ok=True)
    setting_list = {
        'text_prompts': text_prompts,
        'user_comment': user_comment,
        'image_prompts': image_prompts,
        'range_scale': range_scale,
        'sat_scale': sat_scale,
        'max_frames': max_frames,
        'interp_spline': interp_spline,
        'init_image': init_image,
        'clamp_grad': clamp_grad,
        'clamp_max': clamp_max,
        'seed': seed,
        'width': width_height[0],
        'height': width_height[1],
        'diffusion_model': diffusion_model,
        'diffusion_steps': diffusion_steps,
        'max_frames': max_frames,
        'video_init_path': video_init_path,
        'extract_nth_frame': extract_nth_frame,
        'flow_video_init_path': flow_video_init_path,
        'flow_extract_nth_frame': flow_extract_nth_frame,
        'video_init_seed_continuity': video_init_seed_continuity,
        'turbo_mode': turbo_mode,
        'turbo_steps': turbo_steps,
        'turbo_preroll': turbo_preroll,
        'flow_warp': flow_warp,
        'check_consistency': check_consistency,
        'turbo_frame_skips_steps': turbo_frame_skips_steps,
        'forward_weights_clip': forward_weights_clip,
        'forward_weights_clip_turbo_step': forward_weights_clip_turbo_step,
        'padding_ratio': padding_ratio,
        'padding_mode': padding_mode,
        'consistency_blur': consistency_blur,
        'inpaint_blend': inpaint_blend,
        'match_color_strength': match_color_strength,
        'high_brightness_threshold': high_brightness_threshold,
        'high_brightness_adjust_ratio': high_brightness_adjust_ratio,
        'low_brightness_threshold': low_brightness_threshold,
        'low_brightness_adjust_ratio': low_brightness_adjust_ratio,
        'stop_early': stop_early,
        'high_brightness_adjust_fix_amount': high_brightness_adjust_fix_amount,
        'low_brightness_adjust_fix_amount': low_brightness_adjust_fix_amount,
        'max_brightness_threshold': max_brightness_threshold,
        'min_brightness_threshold': min_brightness_threshold,
        'enable_adjust_brightness': enable_adjust_brightness,
        'dynamic_thresh': dynamic_thresh,
        'warp_interp': warp_interp,
        'fixed_code': fixed_code,
        'code_randomness': code_randomness,
        # 'normalize_code': normalize_code,
        'mask_result': mask_result,
        'reverse_cc_order': reverse_cc_order,
        'flow_lq': flow_lq,
        'use_predicted_noise': use_predicted_noise,
        'clip_guidance_scale': clip_guidance_scale,
        'clip_type': clip_type,
        'clip_pretrain': clip_pretrain,
        'missed_consistency_weight': missed_consistency_weight,
        'overshoot_consistency_weight': overshoot_consistency_weight,
        'edges_consistency_weight': edges_consistency_weight,
        'style_strength_schedule': style_strength_schedule_bkup,
        'flow_blend_schedule': flow_blend_schedule_bkup,
        'steps_schedule': steps_schedule_bkup,
        'init_scale_schedule': init_scale_schedule_bkup,
        'latent_scale_schedule': latent_scale_schedule_bkup,
        'latent_scale_template': latent_scale_template,
        'init_scale_template': init_scale_template,
        'steps_template': steps_template,
        'style_strength_template': style_strength_template,
        'flow_blend_template': flow_blend_template,
        'make_schedules': make_schedules,
        'normalize_latent': normalize_latent,
        'normalize_latent_offset': normalize_latent_offset,
        'colormatch_frame': colormatch_frame,
        'use_karras_noise': use_karras_noise,
        'end_karras_ramp_early': end_karras_ramp_early,
        'use_background_mask': use_background_mask,
        'apply_mask_after_warp': apply_mask_after_warp,
        'background': background,
        'background_source': background_source,
        'mask_source': mask_source,
        'extract_background_mask': extract_background_mask,
        'mask_video_path': mask_video_path,
        'negative_prompts': negative_prompts,
        'invert_mask': invert_mask,
        'warp_strength': warp_strength,
        'flow_override_map': flow_override_map,
        'cfg_scale_schedule': cfg_scale_schedule_bkup,
        'respect_sched': respect_sched,
        'color_match_frame_str': color_match_frame_str,
        'colormatch_offset': colormatch_offset,
        'latent_fixed_mean': latent_fixed_mean,
        'latent_fixed_std': latent_fixed_std,
        'colormatch_method': colormatch_method,
        'colormatch_regrain': colormatch_regrain,
        'warp_mode': warp_mode,
        'use_patchmatch_inpaiting': use_patchmatch_inpaiting,
        'blend_latent_to_init': blend_latent_to_init,
        'warp_towards_init': warp_towards_init,
        'init_grad': init_grad,
        'grad_denoised': grad_denoised,
        'colormatch_after': colormatch_after,
        'colormatch_turbo': colormatch_turbo,
        'model_version': model_version,
        'cond_image_src': cond_image_src,
        'warp_num_k': warp_num_k,
        'warp_forward': warp_forward,
        'sampler': sampler.__name__,
        'mask_clip': (mask_clip_low, mask_clip_high),
        'inpainting_mask_weight': inpainting_mask_weight,
        'inverse_inpainting_mask': inverse_inpainting_mask,
        'mask_source': mask_source,
        'model_path': model_path,
        'diff_override': diff_override,
        'image_scale_schedule': image_scale_schedule_bkup,
        'image_scale_template': image_scale_template,
        'frame_range': frame_range,
        'detect_resolution': detect_resolution,
        'bg_threshold': bg_threshold,
        'diffuse_inpaint_mask_blur': diffuse_inpaint_mask_blur,
        'diffuse_inpaint_mask_thresh': diffuse_inpaint_mask_thresh,
        'add_noise_to_latent': add_noise_to_latent,
        'noise_upscale_ratio': noise_upscale_ratio,
        'fixed_seed': fixed_seed,
        'init_latent_fn': init_latent_fn.__name__,
        'value_threshold': value_threshold,
        'distance_threshold': distance_threshold,
        'masked_guidance': masked_guidance,
        'cc_masked_diffusion': cc_masked_diffusion,
        'alpha_masked_diffusion': alpha_masked_diffusion,
        'inverse_mask_order': inverse_mask_order,
        'invert_alpha_masked_diffusion': invert_alpha_masked_diffusion,
        'quantize': quantize,
        'cb_noise_upscale_ratio': cb_noise_upscale_ratio,
        'cb_add_noise_to_latent': cb_add_noise_to_latent,
        'cb_use_start_code': cb_use_start_code,
        'cb_fixed_code': cb_fixed_code,
        'cb_norm_latent': cb_norm_latent,
        'guidance_use_start_code': guidance_use_start_code,
        'offload_model': offload_model,
        'controlnet_preprocess': controlnet_preprocess,
        'small_controlnet_model_path': small_controlnet_model_path,
        'use_scale': use_scale,
        'g_invert_mask': g_invert_mask,
        'controlnet_multimodel': json.dumps(controlnet_multimodel),
        'img_zero_uncond': img_zero_uncond,
        'do_softcap': do_softcap,
        'softcap_thresh': softcap_thresh,
        'softcap_q': softcap_q,
        'deflicker_latent_scale': deflicker_latent_scale,
        'deflicker_scale': deflicker_scale,
        'controlnet_multimodel_mode': controlnet_multimodel_mode,
        'no_half_vae': no_half_vae,
        'temporalnet_source': temporalnet_source,
        'temporalnet_skip_1st_frame': temporalnet_skip_1st_frame,
        'rec_randomness': rec_randomness,
        'rec_source': rec_source,
        'rec_cfg': rec_cfg,
        'rec_prompts': rec_prompts,
        'inpainting_mask_source': inpainting_mask_source,
        'rec_steps_pct': rec_steps_pct,
        'max_faces': max_faces,
        'num_flow_updates': num_flow_updates,
        'control_sd15_openpose_hands_face': control_sd15_openpose_hands_face,
        'control_sd15_depth_detector': control_sd15_depth_detector,
        'control_sd15_softedge_detector': control_sd15_softedge_detector,
        'control_sd15_seg_detector': control_sd15_seg_detector,
        'control_sd15_scribble_detector': control_sd15_scribble_detector,
        'control_sd15_lineart_coarse': control_sd15_lineart_coarse,
        'control_sd15_inpaint_mask_source': control_sd15_inpaint_mask_source,
        'control_sd15_shuffle_source': control_sd15_shuffle_source,
        'control_sd15_shuffle_1st_source': control_sd15_shuffle_1st_source,
        'overwrite_rec_noise': overwrite_rec_noise,
        'use_legacy_cc': use_legacy_cc,
        'missed_consistency_dilation': missed_consistency_dilation,
        'edge_consistency_width': edge_consistency_width,
        'use_reference': use_reference,
        'reference_weight': reference_weight,
        'reference_source': reference_source,
        'reference_mode': reference_mode,
        'use_legacy_fixed_code': use_legacy_fixed_code,
        'consistency_dilate': consistency_dilate,
        'prompt_patterns_sched': prompt_patterns_sched
    }
    if not skip_save:
        try:
            settings_fname = f"{settings_out}/{batch_name}({batchNum})_settings.txt"
            if os.path.exists(settings_fname):
                s_meta = os.path.getmtime(settings_fname)
                os.rename(settings_fname, settings_fname[:-4] + str(s_meta) + '.txt')
            with open(settings_fname, "w+") as f:  # save settings
                json.dump(setting_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(e)
            print('Settings:', setting_list)
    return setting_list
