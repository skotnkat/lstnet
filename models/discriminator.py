import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import Conv2dExtended, MaxPool2dExtended

CLF_THRESHOLD = 0.5
EPSILON = 1e-8


class Discriminator(LstnetComponent):
    def __init__(self, input_size, in_channels_num, params, negative_slope=0.01):
        self.dense_layer_params = params[-1]
        self.leaky_relu_neg_slope = negative_slope

        self.last_layer_idx = 2

        # pass all the params apart from leaky relu and last layer
        super().__init__(input_size, in_channels_num, params[:-1])

        last_output_size = self.get_last_layer_output_size()  
        last_layer_out_channels = self.get_last_layer_out_channels()

        in_features = last_output_size[0]*last_output_size[1]*last_layer_out_channels

        self.dense_layer_params["in_features"] = in_features
        last_layer = self._create_last_layer()

        self.layers.append(last_layer)

    def forward(self, x):    
        x = self.layers.forward(x)
        return x  # returns raw

    def _create_stand_layer(self, params, in_channels, input_size):
        conv_params, pool_params = params
        conv = Conv2dExtended(in_channels, input_size=input_size, **conv_params)  
        relu = nn.LeakyReLU(negative_slope=self.leaky_relu_neg_slope)

        output_size = conv.compute_output_size(input_size)

        pool = MaxPool2dExtended(input_size=output_size, **pool_params)
        
        layer = nn.Sequential(conv, relu, pool)
    
        return layer

    def _create_last_layer(self):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(**self.dense_layer_params),
        )
    
        return last_layer

    @staticmethod
    def _compute_layer_output_size(layer, input_size):
        conv_output_size = layer[0].compute_output_size(input_size)   
        
        pool_output_size = layer[2].compute_output_size(conv_output_size)        

        return pool_output_size
