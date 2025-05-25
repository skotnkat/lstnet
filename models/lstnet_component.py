from abc import ABC, abstractmethod
import torch.nn as nn


class LstnetComponent(ABC, nn.Module):
    def __init__(self, input_size, in_channels_num, params, skip_last_layer=False):
        super().__init__()
        
        self.layers = nn.Sequential()
        self.output_sizes = []
        self.params = params

        stand_layers_num = len(self.params) - skip_last_layer
        for i in range(stand_layers_num):
            # passing only convolution
            output_size = self._add_layer(input_size, in_channels_num, i, self._create_stand_layer)
            
            input_size = output_size
            in_channels_num = self.get_last_layer_out_channels()

    def forward(self, x):
        x = self.layers.forward(x)
        return x

    def _add_layer(self, input_size, in_channels_num, idx, create_func):
        layer = create_func(self.params[idx], in_channels_num, input_size=input_size)
        self.layers.add_module(name=f'layer_{idx}', module=layer)

        output_size = self._compute_layer_output_size(layer, input_size)
        self.output_sizes.append(output_size)

        return output_size

    @abstractmethod
    def _create_stand_layer(self, params, in_channels, input_size=None):
        pass

    @staticmethod
    @abstractmethod
    def _compute_layer_output_size(layer, input_size):
        pass    

    def get_last_layer_out_channels(self):
        return self.layers[-1][0].out_channels
    
    def get_last_layer_output_size(self):
        return self.output_sizes[-1]
