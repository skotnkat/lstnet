import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import Conv2dExtended


class Encoder(LstnetComponent):
    def __init__(self, input_size, in_channels_num, params,
                 negative_slope=0.01, momentum=0.1, **kwargs):
        self.leaky_relu_negative_slope = negative_slope
        self.batch_norm_momentum = momentum

        super().__init__(input_size, in_channels_num, params)

    def _create_stand_layer(self, params, in_channels, input_size):
        layer = nn.Sequential(
            Conv2dExtended(in_channels, input_size=input_size, **params),
            nn.BatchNorm2d(params["out_channels"], momentum=self.batch_norm_momentum),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope)
        )

        return layer

    @staticmethod
    def _compute_layer_output_size(layer, input_size):
        return layer[0].compute_output_size(input_size)
