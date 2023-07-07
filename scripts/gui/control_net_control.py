from ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, \
    SelectionSlider, Valid
class ControlNetControls(HBox):
    def __init__(self, name, values, **kwargs):
        self.label = HTML(
            description=name,
            description_tooltip=name, style={'description_width': 'initial'},
            layout=Layout(position='relative', left='-25px', width='200px'))

        self.enable = Checkbox(value=values['weight'] > 0, description='', indent=True, description_tooltip='Enable model.',
                               style={'description_width': '25px'}, layout=Layout(width='70px', left='-25px'))
        self.weight = FloatText(value=values['weight'], description=' ', step=0.05,
                                description_tooltip='Controlnet model weights. ', layout=Layout(width='100px', visibility='visible' if values['weight'] > 0 else 'hidden'),
                                style={'description_width': '25px'})
        self.start_end = FloatRangeSlider(
            value=[values['start'], values['end']],
            min=0,
            max=1,
            step=0.01,
            description=' ',
            description_tooltip='Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            layout=Layout(width='300px', visibility='visible' if values['weight'] > 0 else 'hidden'),
            style={'description_width': '50px'}
        )

        self.enable.observe(self.on_change)
        self.weight.observe(self.on_change)

        super().__init__([self.enable, self.label, self.weight, self.start_end], layout=Layout(valign='center'))

    def on_change(self, change):
        # print(change)
        if change['name'] == 'value':

            if self.enable.value:
                self.weight.disabled = False
                self.weight.layout.visibility = 'visible'
                if change['old'] == False and self.weight.value == 0:
                    self.weight.value = 1
                # if self.weight.value>0:
                self.start_end.disabled = False
                self.label.disabled = False
                self.start_end.layout.visibility = 'visible'
            else:
                self.weight.disabled = True
                self.start_end.disabled = True
                self.label.disabled = True
                self.weight.layout.visibility = 'hidden'
                self.start_end.layout.visibility = 'hidden'

    def __getattr__(self, attr):
        if attr == 'value':
            weight = 0
            if self.weight.value > 0 and self.enable.value: weight = self.weight.value
            (start, end) = self.start_end.value
            return {
                "weight": weight,
                "start": start,
                "end": end
            }
        else:
            return super.__getattr__(attr)
