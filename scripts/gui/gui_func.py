import os
from glob import glob
import PIL
from ipywidgets import HTML, Layout, HBox, Textarea, Checkbox

import k_diffusion
from scripts.gui.control_gui import ControlGUI
from scripts.run.do_and_run import settings_out
from scripts.settings.setting import batchFolder

def set_visibility(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].layout.visibility = value

def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key].value
        else:
            for o in obj.keys():
                res = get_value(key, obj[o])
                if res is not None: return res
    if isinstance(obj, list):
        for o in obj:
            res = get_value(key, o)
            if res is not None: return res
    return None

def set_value(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].value = value
        else:
            for o in obj.keys():
                set_value(key, value, obj[o])

    if isinstance(obj, list):
        for o in obj:
            set_value(key, value, o)

def add_labels_dict(gui):
    style = {'description_width': '250px'}
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        gui[key].style = style
        # temp = gui[key]
        # temp.observe(dump_gui())
        # gui[key] = temp
        if isinstance(gui[key], ControlGUI):
            continue
        if not isinstance(gui[key], Textarea) and not isinstance(gui[key], Checkbox):
            # vis = gui[key].layout.visibility
            # gui[key].layout = layout
            gui[key].layout.width = '500px'
        if isinstance(gui[key], Checkbox):
            html_label = HTML(
                description=gui[key].description,
                description_tooltip=gui[key].description_tooltip, style={'description_width': 'initial'},
                layout=Layout(position='relative', left='-25px'))
            gui_labels[key] = HBox([gui[key], html_label])
            gui_labels[key].layout.visibility = gui[key].layout.visibility
            gui[key].description = ''
            # gui_labels[key] = gui[key]

        else:

            gui_labels[key] = gui[key]
            # gui_labels[key].layout.visibility = gui[key].layout.visibility
        # gui_labels[key].observe(print('smth changed', time.time()))

    return gui_labels

import json

def infer_settings_path(path):
    default_settings_path = path
    if default_settings_path == '-1':
        settings_files = sorted(glob(os.path.join(batchFolder + f"/settings", '*.txt')))
        if len(settings_files) > 0:
            default_settings_path = settings_files[-1]
        else:
            print('Skipping load latest run settings: no settings files found.')
            return ''
    else:
        try:
            if type(eval(default_settings_path)) == int:
                files = sorted(glob(os.path.join(settings_out, '*.txt')))
                for f in files:
                    if f'({default_settings_path})' in f:
                        default_settings_path = f
        except:
            pass

    path = default_settings_path
    return path

def load_settings(path, default_settings_path):
    path = infer_settings_path(path)

    global guis, load_settings_path, output
    if not os.path.exists(path):
        output.clear_output()
        print('Please specify a valid path to a settings file.')
        return
    if path.endswith('png'):
        img = PIL.Image.open(path)
        exif_data = img._getexif()
        settings = json.loads(exif_data[37510])

    else:
        print('Loading settings from: ', default_settings_path)
        with open(path, 'rb') as f:
            settings = json.load(f)

    for key in settings:
        try:
            val = settings[key]
            if key == 'normalize_latent' and val == 'first_latent':
                val = 'init_frame'
                settings['normalize_latent_offset'] = 0
            if key == 'turbo_frame_skips_steps' and val == None:
                val = '100% (don`t diffuse turbo frames, fastest)'
            if key == 'seed':
                key = 'set_seed'
            if key == 'grad_denoised ':
                key = 'grad_denoised'
            if type(val) in [dict, list]:
                if type(val) in [dict]:
                    temp = {}
                    for k in val.keys():
                        temp[int(k)] = val[k]
                    val = temp
                val = json.dumps(val)
            if key == 'mask_clip':
                val = eval(val)
            if key == 'sampler':
                val = getattr(k_diffusion.sampling, val)
            if key == 'controlnet_multimodel':
                val = val.replace('control_sd15_hed', 'control_sd15_softedge')
                val = json.loads(val)
            # print(key, val)
            set_value(key, val, guis)
            # print(get_value(key, guis))
        except Exception as e:
            print(key), print(settings[key])
            print(e)
    # output.clear_output()
    print('Successfully loaded settings from ', path)

def dump_gui():
    import time
    print('smth changed', time.time())
