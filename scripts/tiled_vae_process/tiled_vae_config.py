from dataclasses import dataclass
@dataclass
class TiledVaeConfig:
    use_tiled_vae = False  # @param {'type':'boolean'}
    tile_size = 128  # \@param {'type':'number'}
    stride = 96  # \@param {'type':'number'}
    num_tiles = [2, 2]  # @param {'type':'raw'}
    padding = [0.5, 0.5]  # \@param {'type':'raw'}
    if num_tiles in [0, '', None]:
        num_tiles = None
    if padding in [0, '', None]:
        padding = None