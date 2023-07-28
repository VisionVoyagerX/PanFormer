import torch
from torch import nn

from .common.modules import conv3x3, SwinModule
import matplotlib.pyplot as plt


class CrossSwinTransformer(nn.Module):
    def __init__(self, ms_channels=4, n_feats=64, n_heads=4, head_dim=8, win_size=4,
                 n_blocks=3, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False, **kwargs):
        super().__init__()
        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std = kwargs.get('mslr_std')
        self.pan_mean = kwargs.get('pan_mean')
        self.pan_std = kwargs.get('pan_std')

        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False)
        ]

        ms_encoder = [
            SwinModule(in_channels=ms_channels, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False)
        ]

        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(n_blocks):
                self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            ms_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                         downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                         window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            pan_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                          downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                          window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        self.HR_tail = nn.Sequential(
            conv3x3(n_feats * len(cat_feat), n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, ms_channels))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

    def forward(self, pan, ms):
        # channel-wise normalization
        # pan = (pan - self.pan_mean) / self.pan_std
        # ms = (ms - self.mslr_mean) / self.mslr_std
        pan = data_normalize(pan, 10)
        ms = data_normalize(ms, 10)

        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms)

        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        for i in range(self.n_blocks):
            if 'pan' in self.cross_module:
                pan_cross_ms_feat = self.pan_cross_ms[i](
                    last_pan_feat, last_ms_feat)
            if 'ms' in self.cross_module:
                ms_cross_pan_feat = self.ms_cross_pan[i](
                    last_ms_feat, last_pan_feat)
            if 'pan' in self.cross_module:
                last_pan_feat = pan_cross_ms_feat
            if 'ms' in self.cross_module:
                last_ms_feat = ms_cross_pan_feat

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(last_pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(last_ms_feat)

        output = self.HR_tail(torch.cat(cat_list, dim=1))

        output = torch.clamp(output, 0, 2 ** 10 - .5)

        # output = output * self.mslr_std + self.mslr_mean
        output = data_denormalize(output, 10)

        return output


def data_normalize(img, bit_depth):
    """ Normalize the data to [0, 1)

    Args:
        img_dict (dict[str, torch.Tensor]): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: images after normalization
    """
    max_value = 2 ** bit_depth - .5

    img = img / max_value
    return img


def data_denormalize(img, bit_depth):
    """ Denormalize the data to [0, n-bit)

    Args:
        img (torch.Tensor | np.ndarray): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: image after denormalize
    """
    max_value = 2 ** bit_depth - .5
    ret = img * max_value
    return ret


if __name__ == "__main__":
    crossswintransformer = CrossSwinTransformer()
    lr = torch.randn(1, 4, 64, 64)
    pan = torch.rand(1, 1, 256, 256)

    print(crossswintransformer(pan, lr).shape)
