# from dataclasses import dataclass
#
# from scripts.settings.setting import batchFolder
#
# @dataclass
# class GUIConfig:
#     gui_difficulty = "Hey, not too rough."  # @param ["I'm too young to die.", "Hey, not too rough.", "Ultra-Violence."]
#     print(f'Using "{gui_difficulty}" gui difficulty. Please switch to another difficulty\nto unlock up to {len(gui_difficulty_dict[gui_difficulty])} more settings when you`re ready :D')
#     default_settings_path = ''  # @param {'type':'string'}
#     load_default_settings = True  # @param {'type':'boolean'}
#     # @markdown Disable to load settings into GUI from colab cells. You will need to re-run colab cells you've edited to apply changes, then re-run the gui cell.\
#     # @markdown Enable to keep GUI state.
#     keep_gui_state_on_cell_rerun = True  # @param {'type':'boolean'}
#     settings_out = batchFolder + f"/settings"