import torch.nn as nn
import torch.nn.functional as F
from utils import LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING
import utils

expected_params = [LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING]


class Generator(nn.Module):
    def __init__(self, in_channels_num, params, shared_encoder=None, last_layer=False):
        super().__init__()
        self.layers = None
        self.to_pad = []

        utils.check_and_transform_params(params, expected_params)

        for i, layer in enumerate(shared_encoder.layers):
            self.layers.append(layer)
            self.to_pad.append(shared_encoder.to_pad[i])

        for i in range(params[LAYERS_NUM]):
            out_channels_num = params[OUT_CHANNELS_NUM][i]
            kernel_size = params[KERNEL_SIZE][i]
            stride = params[STRIDE][i]
            padding = params[PADDING][i]

            self.layers.append(get_stand_generator_layer(in_channels_num, out_channels_num, kernel_size, stride))
            self.to_pad.append(padding == "same")  # store if we need to pad the x data

            in_channels_num = out_channels_num

        if last_layer:
            self.layers.append(get_last_generator_layer(out_channels_num))
            self.to_pad.append(False)

    def forward(self, x):
        print(f'Start: {x.shape}')
        for i, layer in enumerate(self.layers):
            pad_input_data(x, self.to_pad[i], layer[0].stride)
            x = self.layers(x)
            print(f'{i}th layer: {x.shape}')
        print(f'End: {x.shape}')

        return x
    

def get_stand_generator_layer(in_channels, out_channels, kernel_size, stride, padding):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )
    
    return layer


def get_last_generator_layer(in_channels):
    last_layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),  # change the padding, shoudl be same
        nn.Tanh()            
    )

    return last_layer


# can pass function and have the main one in utility
def compute_padding(kernel_size, stride):
    pad_width = kernel_size[0] - stride[0]
    pad_height = kernel_size[1] - stride[1]

    # split padding to directions
    pad_left = pad_width[0] // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    return pad_left, pad_right, pad_top, pad_bottom


def pad_input_data(x, to_pad, kernel_size, stride):
    if not to_pad:
        return x

    padding = compute_padding(kernel_size, stride)
    x = F.pad(x, padding)

    return x



