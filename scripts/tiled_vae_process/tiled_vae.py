# @title Tiled VAE
# @markdown Enable if you're getting CUDA Out of memory errors during encode_first_stage or decode_fiirst_stage.
# @markdown Is slower.
# tiled vae from thttps://github.com/CompVis/latent-diffusion
import math
import time

import torch
from einops import rearrange

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from scripts.tiled_vae_process.tiled_vae_config import TiledVaeConfig

def tiled_vae(config: TiledVaeConfig, sd_model):
    def get_fold_unfold(x, kernel_size, stride, uf=1, df=1, self=sd_model):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[1] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[1] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        normalization = torch.where(normalization == 0., 1e-6, normalization)
        return fold, unfold, normalization, weighting

    # non divisible by 8 fails here

    @torch.no_grad()
    def encode_first_stage(x, self=sd_model):
        ts = time.time()
        if hasattr(self, "split_input_params"):

            if self.split_input_params["patch_distributed_vq"]:
                print('------using tiled vae------')
                bs, nc, h, w = x.shape
                df = self.split_input_params["vqf"]
                if self.split_input_params["num_tiles"] is not None:
                    num_tiles = self.split_input_params["num_tiles"]
                    ks = [h // num_tiles[0], w // num_tiles[1]]
                else:
                    ks = self.split_input_params["ks"]  # eg. (128, 128)
                    ks = [o * (df) for o in ks]
                # ks = self.split_input_params["ks"]  # eg. (128, 128)
                # ks = [o*df for o in ks]

                if self.split_input_params["padding"] is not None:
                    padding = self.split_input_params["padding"]
                    stride = [int(ks[0] * padding[0]), int(ks[1] * padding[1])]
                else:
                    stride = self.split_input_params["stride"]  # eg. (64, 64)
                    stride = [o * (df) for o in stride]
                # stride = self.split_input_params["stride"]  # eg. (64, 64)
                # stride = [o*df for o in stride]
                # ks = [512,512]
                # stride = [512,512]

                # print('kernel, stride', ks, stride)

                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape

                target_h = math.ceil(h / ks[0]) * ks[0]
                target_w = math.ceil(w / ks[1]) * ks[1]
                padh = target_h - h
                padw = target_w - w
                pad = (0, padw, 0, padh)
                if target_h != h or target_w != w:
                    print('Padding.')
                    # print('padding from ', h, w, 'to ', target_h, target_w)
                    x = torch.nn.functional.pad(x, pad, mode='reflect')
                    # print('padded from ', h, w, 'to ', z.shape[2:])

                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # print('z', z.shape)
                output_list = [self.get_first_stage_encoding(self.first_stage_model.encode(z[:, :, :, :, i]), tiled_vae_call=True)
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting
                # print('o', o.shape)
                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                print('Tiled vae encoder took ', f'{time.time() - ts:.2}')
                # print('decoded stats', decoded.min(), decoded.max(), decoded.std(), decoded.mean())
                return decoded[..., :h // df, :w // df]

            else:
                print('Vae encoder took ', f'{time.time() - ts:.2}')
                # print('x stats', x.min(), x.max(), x.std(), x.mean())
                return self.first_stage_model.encode(x)
        else:
            print('Vae encoder took ', f'{time.time() - ts:.2}')
            # print('x stats', x.min(), x.max(), x.std(), x.mean())
            return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(z, predict_cids=False, force_not_quantize=False, self=sd_model):
        ts = time.time()
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            print('------using tiled vae------')
            # print('latent shape: ', z.shape)
            # print(self.split_input_params)
            if self.split_input_params["patch_distributed_vq"]:
                bs, nc, h, w = z.shape
                if self.split_input_params["num_tiles"] is not None:
                    num_tiles = self.split_input_params["num_tiles"]
                    ks = [h // num_tiles[0], w // num_tiles[1]]
                else:
                    ks = self.split_input_params["ks"]  # eg. (128, 128)

                if self.split_input_params["padding"] is not None:
                    padding = self.split_input_params["padding"]
                    stride = [int(ks[0] * padding[0]), int(ks[1] * padding[1])]
                else:
                    stride = self.split_input_params["stride"]  # eg. (64, 64)

                uf = self.split_input_params["vqf"]

                target_h = math.ceil(h / ks[0]) * ks[0]
                target_w = math.ceil(w / ks[1]) * ks[1]
                padh = target_h - h
                padw = target_w - w
                pad = (0, padw, 0, padh)
                if target_h != h or target_w != w:
                    print('Padding.')
                    # print('padding from ', h, w, 'to ', target_h, target_w)
                    z = torch.nn.functional.pad(z, pad, mode='reflect')
                    # print('padded from ', h, w, 'to ', z.shape[2:])

                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                # print(ks, stride)
                fold, unfold, normalization, weighting = get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # print('z unfold, normalization, weighting',z.shape, normalization.shape, weighting.shape)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                # print('z unfold view , normalization, weighting',z.shape)
                # 2. apply model loop over last dim
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                # print('out stack', o.shape)

                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                print('Tiled vae decoder took ', f'{time.time() - ts:.2}')
                # print('decoded stats', decoded.min(), decoded.max(), decoded.std(), decoded.mean())
                # assert False
                return decoded[..., :h * uf, :w * uf]
            else:
                print('Vae decoder took ', f'{time.time() - ts:.2}')
                # print('z stats', z.min(), z.max(), z.std(), z.mean())
                return self.first_stage_model.decode(z)

        else:
            # print('z stats', z.min(), z.max(), z.std(), z.mean())
            print('Vae decoder took ', f'{time.time() - ts:.2}')
            return self.first_stage_model.decode(z)

    def get_first_stage_encoding(encoder_posterior, self=sd_model, tiled_vae_call=False):
        if hasattr(self, "split_input_params") and not tiled_vae_call:
            # pass for tiled vae
            return encoder_posterior

        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    bkup_decode_first_stage = sd_model.decode_first_stage
    bkup_encode_first_stage = sd_model.encode_first_stage
    bkup_get_first_stage_encoding = sd_model.get_first_stage_encoding
    bkup_get_fold_unfold = sd_model.get_fold_unfold
    if config.use_tiled_vae:
        ks = config.tile_size
        stride = config.stride
        vqf = 8  #
        split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                              "num_tiles": config.num_tiles, "padding": config.padding,
                              "vqf": vqf,
                              "patch_distributed_vq": True,
                              "tie_braker": False,
                              "clip_max_weight": 0.5,
                              "clip_min_weight": 0.01,
                              "clip_max_tie_weight": 0.5,
                              "clip_min_tie_weight": 0.01}

        sd_model.split_input_params = split_input_params
        sd_model.decode_first_stage = decode_first_stage
        sd_model.encode_first_stage = encode_first_stage
        sd_model.get_first_stage_encoding = get_first_stage_encoding
        sd_model.get_fold_unfold = get_fold_unfold

    else:
        if hasattr(sd_model, "split_input_params"):
            delattr(sd_model, "split_input_params")
            try:
                sd_model.decode_first_stage = bkup_decode_first_stage
                sd_model.encode_first_stage = bkup_encode_first_stage
                sd_model.get_first_stage_encoding = bkup_get_first_stage_encoding
                sd_model.get_fold_unfold = bkup_get_fold_unfold
            except:
                pass
