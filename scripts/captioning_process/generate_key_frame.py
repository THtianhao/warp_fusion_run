# @title Generate captions for keyframes
# @markdown Automatically generate captions for every n-th frame, \
# @markdown or keyframe list: at keyframe, at offset from keyframe, between keyframes.\
# @markdown keyframe source: Every n-th frame, user-input, Content-aware scheduling keyframes
import os
import pathlib
from glob import glob

from scripts.captioning_process.captioning_config import CaptioningConfig
from scripts.utils.path import createPath

def load_caption(caption_file):
    caption = ''
    with open(caption_file, 'r') as f:
        caption = f.read()
    return caption

def get_caption(frame_num, videoFramesCaptions):
    caption_files = sorted(glob(os.path.join(videoFramesCaptions, '*.txt')))
    frame_num1 = frame_num + 1
    if len(caption_files) == 0:
        return None
    frame_numbers = [int(o.replace('\\', '/').split('/')[-1][:-4]) for o in caption_files]
    # print(frame_numbers, frame_num)
    if frame_num1 < frame_numbers[0]:
        return load_caption(caption_files[0])
    if frame_num1 >= frame_numbers[-1]:
        return load_caption(caption_files[-1])
    for i in range(len(frame_numbers)):
        if frame_num1 >= frame_numbers[i] and frame_num1 < frame_numbers[i + 1]:
            return load_caption(caption_files[i])
    return None

def generate_key_frame(config: CaptioningConfig, videoFramesFolder, diff):
    inputFrames = sorted(glob(f'{videoFramesFolder}/*.jpg'))
    keyframes = None
    caption_keyframes = None
    if config.keyframe_source == 'Content-aware scheduling keyframes':
        if diff in [None, '', []]:
            print('ERROR: Keyframes were not generated. Please go back to Content-aware scheduling cell, enable analyze_video nad run it or choose a different caption keyframe source.')
            caption_keyframes = None
        else:
            caption_keyframes = [1] + [i + 1 for i, o in enumerate(diff) if o >= config.diff_thresh]
    if config.keyframe_source == 'User-defined keyframe list':
        caption_keyframes = config.user_defined_keyframes
    if config.keyframe_source == 'Every n-th frame':
        caption_keyframes = list(range(1, len(inputFrames), config.nth_frame))

    config.videoFramesCaptions = videoFramesFolder + 'Captions'
    if config.make_captions and caption_keyframes is not None:
        try:
            config.blip_model
        except:

            os.chdir('./BLIP')
            from models.blip import blip_decoder

            os.chdir('../')
            from PIL import Image
            import torch
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_size = 384
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'  # -O /content/model_base_caption_capfilt_large.pth'

            blip_model = blip_decoder(pretrained=model_url, image_size=384, vit='base', med_config='./BLIP/configs/med_config.json')
            blip_model.eval()
            blip_model = blip_model.to(device)
        finally:
            print('Using keyframes: ', caption_keyframes[:20], ' (first 20 keyframes displyed')
            if offset_mode == 'None':
                keyframes = caption_keyframes
            if offset_mode == 'Fixed':
                keyframes = caption_keyframes
                for i in range(len(caption_keyframes)):
                    if keyframes[i] >= max(caption_keyframes):
                        keyframes[i] = caption_keyframes[i]
                    else:
                        keyframes[i] = min(caption_keyframes[i] + fixed_offset, caption_keyframes[i + 1])
                print('Remapped keyframes to ', keyframes[:20])
            if offset_mode == 'Between Keyframes':
                keyframes = caption_keyframes
                for i in range(len(caption_keyframes)):
                    if keyframes[i] >= max(caption_keyframes):
                        keyframes[i] = caption_keyframes[i]
                    else:
                        keyframes[i] = caption_keyframes[i] + int((caption_keyframes[i + 1] - caption_keyframes[i]) / 2)
                print('Remapped keyframes to ', keyframes[:20])

            videoFramesCaptions = videoFramesFolder + 'Captions'
            createPath(videoFramesCaptions)

        from tqdm.notebook import trange, tqdm

        for f in pathlib.Path(videoFramesCaptions).glob('*.txt'):
            f.unlink()
        for i in tqdm(keyframes):
            with torch.no_grad():
                keyFrameFilename = inputFrames[i - 1]
                raw_image = Image.open(keyFrameFilename)
                image = transform(raw_image).unsqueeze(0).to(device)
                caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=30, min_length=5)
                captionFilename = os.path.join(videoFramesCaptions, keyFrameFilename.replace('\\', '/').split('/')[-1][:-4] + '.txt')
                with open(captionFilename, 'w') as f:
                    f.write(caption[0])
