# from ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, \
#     SelectionSlider, Valid
# class FilePath(HBox):
#     def __init__(self, **kwargs):
#         self.model_path = Text(value='', continuous_update=True, **kwargs)
#         self.path_checker = Valid(
#             value=False, layout=Layout(width='2000px')
#         )
#
#         self.model_path.observe(self.on_change)
#         super().__init__([self.model_path, self.path_checker])
#
#     def __getattr__(self, attr):
#         if attr == 'value':
#             return self.model_path.value
#         else:
#             return super.__getattr__(attr)
#
#     def on_change(self, change):
#         if change['name'] == 'value':
#             if os.path.exists(change['new']):
#                 self.path_checker.value = True
#                 self.path_checker.description = ''
#             else:
#                 self.path_checker.value = False
#                 self.path_checker.description = 'The file does not exist. Please specify the correct path.'
