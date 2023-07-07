from ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, \
    SelectionSlider, Valid

from scripts.gui.control_net_control import ControlNetControls

class ControlGUI(VBox):
    def __init__(self, args):
        enable_label = HTML(
            description='Enable',
            description_tooltip='Enable', style={'description_width': '50px'},
            layout=Layout(width='40px', left='-50px', ))
        model_label = HTML(
            description='Model name',
            description_tooltip='Model name', style={'description_width': '100px'},
            layout=Layout(width='265px'))
        weight_label = HTML(
            description='weight',
            description_tooltip='Model weight. 0 weight effectively disables the model. The total sum of all the weights will be normalized to 1.', style={'description_width': 'initial'},
            layout=Layout(position='relative', left='-25px', width='125px'))  # 65
        range_label = HTML(
            description='active range (% or total steps)',
            description_tooltip='Model`s active range. % of total steps when the model is active.\n Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
            style={'description_width': 'initial'},
            layout=Layout(position='relative', left='-25px', width='200px'))
        controls_list = [HBox([enable_label, model_label, weight_label, range_label])]
        controls_dict = {}
        self.possible_controlnets = ['control_sd15_depth',
                                     'control_sd15_canny',
                                     'control_sd15_softedge',
                                     'control_sd15_mlsd',
                                     'control_sd15_normalbae',
                                     'control_sd15_openpose',
                                     'control_sd15_scribble',
                                     'control_sd15_seg',
                                     'control_sd15_temporalnet',
                                     'control_sd15_face',
                                     'control_sd15_ip2p',
                                     'control_sd15_inpaint',
                                     'control_sd15_lineart',
                                     'control_sd15_lineart_anime',
                                     'control_sd15_shuffle']
        for key in self.possible_controlnets:
            if key in args.keys():
                w = ControlNetControls(key, args[key])
            else:
                w = ControlNetControls(key, {
                    "weight": 0,
                    "start": 0,
                    "end": 1
                })
            controls_list.append(w)
            controls_dict[key] = w

        self.args = args
        self.ws = controls_dict
        super().__init__(controls_list)

    def __getattr__(self, attr):
        if attr == 'value':
            res = {}
            for key in self.possible_controlnets:
                if self.ws[key].value['weight'] > 0:
                    res[key] = self.ws[key].value
            return res
        else:
            return super.__getattr__(attr)
