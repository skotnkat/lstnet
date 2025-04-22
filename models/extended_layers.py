"""
This module extends the functionality of Conv2d, ConvTranspose2d and MaxPool2d to be able to apply the same logic
of padding="same" as in tensorflow. In pytorch is by default only available option padding="same" only when stride
is equal to 1 as it complicates things because of asymmetric padding.
"""


import torch.nn as nn
import torch.nn.functional as F

import utils


class Conv2dExtended(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=(1, 1), input_size=None, **kwargs):
        tmp_padding, is_padding_same = utils.standardize_padding(padding)
        self.is_padding_same = is_padding_same
        
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=tmp_padding, dilation=dilation, **kwargs)

        self.padding_precomputed = True  # false only for padding="same" when no input size is passed
        self.internal_padding = tmp_padding
        
        if is_padding_same:
            if input_size is None:
                self.padding_precomputed = False
            else:
                self.internal_padding = self._compute_padding(input_size)

    def forward(self, x):
        x = self._pad_input(x)
        x = super().forward(x)
        
        return x

    def _pad_input(self, x):
        if not self.is_padding_same:   # valid or number/tuple
            return x

        padding = self.internal_padding
        if not self.padding_precomputed:
            padding = self._compute_padding(x.shape[2:])  # (batch_size, channels_num, width, height)
        
        x = F.pad(x, padding)
        
        return x
    
    def _compute_padding(self, input_size):        
        expected_output_width = (input_size[0] + self.stride[0] - 1) // self.stride[0]  # math.ceil
        expected_output_height = (input_size[1] + self.stride[1] - 1) // self.stride[1]

        effective_kernel_width = utils.compute_effective_kernel_size(self.kernel_size[0], self.dilation[0])
        effective_kernel_height = utils.compute_effective_kernel_size(self.kernel_size[1], self.dilation[1])
        
        p_total_width = (expected_output_width - 1) * self.stride[0] + effective_kernel_width - input_size[0]
        p_total_height = (expected_output_height - 1) * self.stride[1] + effective_kernel_height - input_size[1]
        
        p_left, p_right = utils.split_padding(p_total_width)
        p_top, p_bottom = utils.split_padding(p_total_height)
        
        return p_left, p_right, p_top, p_bottom

    def compute_output_size(self, input_size):
        p_total_width = self.padding[0]
        p_total_height = self.padding[1]

        if self.is_padding_same:
            padding = self.internal_padding
            if not self.padding_precomputed:
                padding = self._compute_padding(input_size)
            
            p_total_width = padding[0] + padding[1]
            p_total_height = padding[2] + padding[3]

        # math.floor
        output_width = (input_size[0]+p_total_width-self.dilation[0]*(self.kernel_size[0]-1)-1) // self.stride[0] + 1
        output_height = (input_size[1]+p_total_height-self.dilation[1]*(self.kernel_size[1]-1)-1) // self.stride[1] + 1
        
        return output_width, output_height


class ConvTranspose2dExtended(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=(1, 1), output_padding=0, **kwargs):
        tmp_padding, is_padding_same = utils.standardize_padding(padding)
        
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=tmp_padding, dilation=dilation, output_padding=output_padding, **kwargs)

        if is_padding_same:
            self.padding, self.output_padding = self._compute_padding()

    # tensorflow assymetric padding
    def _compute_padding(self):  
        effective_kernel_width = utils.compute_effective_kernel_size(self.kernel_size[0], self.dilation[0])
        effective_kernel_height = utils.compute_effective_kernel_size(self.kernel_size[1], self.dilation[1])

        p_total_width = effective_kernel_width - self.stride[0]
        p_total_height = effective_kernel_height - self.stride[1]

        p_output_width = p_total_width % 2
        p_output_height = p_total_height % 2
        
        p_width = p_total_width // 2 + p_output_width
        p_height = p_total_height // 2 + p_output_height

        return (p_width, p_height), (p_output_width, p_output_height)

    def compute_output_size(self, input_size):
        output_width = (input_size[0]-1)*self.stride[0] - 2*self.padding[0] + self.dilation[0]*(self.kernel_size[0]-1) + self.output_padding[0] + 1
        output_height = (input_size[1]-1)*self.stride[1] - 2*self.padding[1] + self.dilation[1]*(self.kernel_size[1]-1) + self.output_padding[1] + 1

        return output_width, output_height


class MaxPool2dExtended(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding, dilation=(1, 1), input_size=None, **kwargs):
        kernel_size = utils.transform_int_to_tuple(kernel_size)
        stride = utils.transform_int_to_tuple(stride)

        tmp_padding, is_padding_same = utils.standardize_padding(padding)
        self.is_padding_same = is_padding_same   
        super().__init__(kernel_size, stride, padding=tmp_padding, dilation=dilation, **kwargs)

        self.padding_precomputed = True
        self.internal_padding = tmp_padding

        if is_padding_same:
            if input_size is None:
                self.padding_precomputed = False
            else:
                self.internal_padding = self._compute_padding(input_size)

    def forward(self, x):
        x = self._pad_input(x)
        x = super().forward(x)
        
        return x

    def _pad_input(self, x):
        if not self.is_padding_same:   # valid or number/tuple
            return x

        padding = self.internal_padding
        if not self.padding_precomputed:
            padding = self._compute_padding(x.shape[2:])  # (batch_size, channels_num, width, height)

        x = F.pad(x, padding)
        
        return x

    def _compute_padding(self, input_size):
        expected_output_width = (input_size[0] + self.stride[0] - 1) // self.stride[0]
        expected_output_height = (input_size[1] + self.stride[1] - 1) // self.stride[1]

        effective_kernel_width = utils.compute_effective_kernel_size(self.kernel_size[0], self.dilation[0])
        effective_kernel_height = utils.compute_effective_kernel_size(self.kernel_size[1], self.dilation[1])

        p_total_width = max((expected_output_width - 1) * self.stride[0] + effective_kernel_width - input_size[0], 0)
        p_total_height = max((expected_output_height - 1) * self.stride[1] + effective_kernel_height - input_size[1], 0)

        p_left, p_right = utils.split_padding(p_total_width)
        p_top, p_bottom = utils.split_padding(p_total_height)

        return p_left, p_right, p_top, p_bottom

    # should not be computing padding, if it was "valid" or something else (check also for other models)
    def compute_output_size(self, input_size):
        p_total_width = self.padding[0]
        p_total_height = self.padding[1]

        if self.is_padding_same:
            padding = self.internal_padding
            if not self.padding_precomputed:
                padding = self._compute_padding(input_size)

            p_total_width = padding[0] + padding[1]  # 2*padding
            p_total_height = padding[2] + padding[3]  # 2*padding

        # math.floor
        output_width = (input_size[0]+p_total_width-self.dilation[0]*(self.kernel_size[0]-1)-1) // self.stride[0] + 1
        output_height = (input_size[1]+p_total_height-self.dilation[1]*(self.kernel_size[1]-1)-1) // self.stride[1] + 1
        
        return output_width, output_height

    