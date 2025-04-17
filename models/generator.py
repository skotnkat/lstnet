import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import ConvTranspose2dExtended


class Generator(LstnetComponent):
    def __init__(self, input_size, in_channels_num, params):
        super().__init__(input_size, in_channels_num, params, skip_last_layer=True)  # pass all the params apart for last layer

        last_output_size = self.get_last_layer_output_size()
        last_out_channels_num = self.get_last_layer_out_channels()    
        self._add_layer(last_output_size, last_out_channels_num, len(params)-1, create_func=Generator._create_last_layer)  # len(params)-1 = idx of layer
    
    @staticmethod
    def _create_stand_layer(params, in_channels, input_size=None):      
        layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params),
            nn.BatchNorm2d(params["out_channels"]),
            nn.LeakyReLU()
        )
    
        return layer

    @staticmethod
    def _create_last_layer(params, in_channels, input_size=None):
        last_layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params),
            nn.Tanh()            
        )
    
        return last_layer

    @staticmethod
    def _compute_layer_output_size(layer, input_size):
        return layer[0].compute_output_size(input_size)
