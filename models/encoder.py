import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import Conv2dExtended


class Encoder(LstnetComponent):
    def __init__(self, input_size, in_channels_num, params):
        self.negative_slope = params[-1]["leaky_relu_neg_slope"]

        super().__init__(input_size, in_channels_num, params[:-1])

    def _create_stand_layer(self, params, in_channels, input_size):
        layer = nn.Sequential(
            Conv2dExtended(in_channels, input_size=input_size, **params),
            nn.BatchNorm2d(params["out_channels"], momentum=0.01, eps=0.001),
            nn.LeakyReLU(negative_slope=self.negative_slope)
        )
    
        return layer

    @staticmethod
    def _compute_layer_output_size(layer, input_size):
        return layer[0].compute_output_size(input_size)
