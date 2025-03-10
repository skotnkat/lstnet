import torch.nn as nn

from utils import check_and_transform_params, LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING, POOLING_KERNEL_SIZE, POOLING_STRIDE

expected_params = [LAYERS_NUM, OUT_CHANNELS_NUM, KERNEL_SIZE, STRIDE, PADDING, POOLING_KERNEL_SIZE, POOLING_STRIDE]


class Discriminator(nn.Module):
    def __init__(self, in_channels_num, params):
        super().__init__()
        self.layers = None

        check_and_transform_params(params, expected_params)

        for i in range(params[LAYERS_NUM]):
            out_channels_num = params[OUT_CHANNELS_NUM][i]
            
            self.layers.append(get_stand_discriminator_layer(in_channels_num, out_channels_num, params[KERNEL_SIZE][i], params[STRIDE][i],
                                                        params[PADDING][i], params[POOLING_KERNEL_SIZE][i], params[POOLING_STRIDE][i]))
            in_channels_num = out_channels_num

        self.layers.append(get_last_discriminator_layer(out_channels_num))

    def forward(self, x):    
        return self.layers(x)
        

def get_stand_discriminator_layer(in_channels, out_channels, kernel_size, stride, padding, pooling_kernel_size, pooling_stride):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(),
        nn.MaxPool2d(pooling_kernel_size, pooling_stride)
        )

    return layer


def get_last_discriminator_layer(in_channels):
    last_layer = nn.Sequential(
        nn.Linear(in_features=in_channels, out_features=1),
        nn.Sigmoid()            
    )

    return last_layer

    