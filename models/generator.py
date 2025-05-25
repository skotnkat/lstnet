import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import ConvTranspose2dExtended


class Generator(LstnetComponent):
    def __init__(self, input_size, in_channels_num, params):
        self.negative_slope = params[-1]
        # pass all the params apart the ones for the last layer
        super().__init__(input_size, in_channels_num, params[:-1], skip_last_layer=True)

        last_output_size = self.get_last_layer_output_size()
        last_out_channels_num = self.get_last_layer_out_channels()
        last_layer_idx = len(params)-2

        self._add_layer(last_output_size, last_out_channels_num, last_layer_idx, create_func=self._create_last_layer)

    def _create_stand_layer(self, params, in_channels, input_size=None):
        layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params),
            nn.BatchNorm2d(params["out_channels"], momentum=0.01, eps=0.001),
            nn.LeakyReLU(negative_slope=0.3)
        )
    
        return layer

    @staticmethod
    def _create_last_layer(params, in_channels, input_size=None):
        last_layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params),
            nn.Tanh()            
        )

        last_layer._is_last_layer = True
    
        return last_layer

    @staticmethod
    def _compute_layer_output_size(layer, input_size):
        return layer[0].compute_output_size(input_size)
