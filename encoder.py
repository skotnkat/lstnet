import torch.nn as nn
import torch.nn.functional as F
from utils import LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING
import utils

expected_params = [LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING]


class Encoder(nn.Module):
    def __init__(self, in_channels_num, params, shared_layers=[]):
        super().__init__()
        self.layers = []
        self.to_pad = []
        
        utils.check_and_transform_params(params, expected_params)

        for i in range(params[LAYERS_NUM]):
            out_channels_num = params[OUT_CHANNELS_NUM][i]
            kernel_size = params[KERNEL_SIZE][i]
            stride = params[STRIDE][i]
            padding = params[PADDING][i]

            self.layers.append(get_stand_encoder_layer(in_channels_num, out_channels_num, kernel_size, stride))
            self.to_pad.append(padding == "same")  # store if we need to pad the x data
            
            in_channels_num = out_channels_num

        for layer in shared_layers:
            self.layers.append(layer)

    def forward(self, x):
        print(f'Start: {x.shape}')
        for i, layer in enumerate(self.layers):
            pad_input_data(x, self.to_pad[i], layer[0].stride)
            x = layer.forward(x)
            print(f'{i}th layer: {x.shape}')
        print(f'End: {x.shape}')

        return x


def get_stand_encoder_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

    return layer


def compute_padding(input_size, kernel_size, stride):
    out_width = (input_size[0]+kernel_size[0]-1) // stride[0]
    out_height = (input_size[1]+kernel_size[1]-1) // stride[1]

    pad_width = (out_width - 1) * stride[0] + kernel_size[0] - input_size[0]
    pad_height = (out_height - 1) * stride[1] + kernel_size[1] - input_size[1]

    # split padding to directions
    pad_left = pad_width[0] // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    return pad_left, pad_right, pad_top, pad_bottom


def pad_input_data(x, to_pad, kernel_size, stride):
    if not to_pad:
        return x

    padding = compute_padding(x.shape, kernel_size, stride)
    x = F.pad(x, padding)

    return x

