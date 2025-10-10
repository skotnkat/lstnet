"""
Module implements extended versions of common neural network layers with additional features
like 'same' padding similar to TensorFlow. These layers include Conv2dExtended,
ConvTranspose2dExtended, and MaxPool2dExtended, which enhance the standard PyTorch layers
by adding functionality for asymmetric padding and dynamic input sizes.

By default, Pytorch uses 'same' padding only for stride of 1,
while TensorFlow supports it for any stride.
"""

from typing import Union, Tuple, cast, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import utils


class Conv2dExtended(nn.Conv2d):
    """Extended 2D convolution layer mirroring TensorFlow with support for 'same' padding."""

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        input_size: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """Extended 2D convolution layer mirroring TensorFlow with support for 'same' padding.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to 1.
            kernel_size (Union[int, Tuple[int, int]], optional): Size of the convolution kernel.
                Defaults to 3.
            stride (Union[int, Tuple[int, int]], optional): Stride of the convolution.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input.
                Defaults to 1.
            dilation (Union[int, Tuple[int, int]], optional): Dilation factor for the convolution.
                Defaults to 1.
            input_size (Optional[Union[int, Tuple[int, int]]], optional): Size of the input tensor.
                Defaults to None.
        """
        tmp_padding, is_padding_same = utils.standardize_padding(padding)
        self.is_padding_same = is_padding_same

        # Convert attributes that have int values into tuples
        kernel_size_converted = utils.transform_int_to_tuple(kernel_size)
        stride_converted = utils.transform_int_to_tuple(stride)
        dilation_converted = utils.transform_int_to_tuple(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_converted,
            stride=stride_converted,
            padding=tmp_padding,
            dilation=dilation_converted,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # false only for padding="same" when no input size is passed
        self.padding_precomputed = True
        self.internal_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)

        if is_padding_same:
            if input_size is None:
                self.padding_precomputed = False
            else:
                self.internal_padding = self._compute_padding(
                    utils.transform_int_to_tuple(input_size)
                )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for the extended convolution layer.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Output tensor after applying the convolution.
        """

        padded_input = self._pad_input(input)
        return super().forward(padded_input)

    def _pad_input(self, x: Tensor) -> Tensor:
        """Pads the input tensor for the convolution operation.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Padded input tensor.
        """

        if not self.is_padding_same:  # valid or number/tuple
            return x

        padding = self.internal_padding
        if not self.padding_precomputed:
            # (N, C, H, W)
            height, width = x.shape[-2], x.shape[-1]
            padding = self._compute_padding((height, width))

        x = F.pad(x, padding)

        return x

    def _compute_padding(
        self, input_size: Tuple[int, int]  # expects (H, W)
    ) -> Tuple[int, int, int, int]:
        """Computes the padding needed for the convolution operation.

        Args:
            input_size (Tuple[int, int]): Size of the input tensor (H, W).

        Returns:
            Tuple[int, int, int, int]: Padding values (left, right, top, bottom).
        """

        expected_output_height = (input_size[0] + self.stride[0] - 1) // self.stride[
            0
        ]  # math.ceil
        expected_output_width = (input_size[1] + self.stride[1] - 1) // self.stride[1]

        effective_kernel_height = utils.compute_effective_kernel_size(
            self.kernel_size[0], self.dilation[0]
        )
        effective_kernel_width = utils.compute_effective_kernel_size(
            self.kernel_size[1], self.dilation[1]
        )

        p_total_height = (
            (expected_output_height - 1) * self.stride[0]
            + effective_kernel_height
            - input_size[0]
        )
        p_total_width = (
            (expected_output_width - 1) * self.stride[1]
            + effective_kernel_width
            - input_size[1]
        )

        p_left, p_right = utils.split_padding(p_total_width)
        p_top, p_bottom = utils.split_padding(p_total_height)

        return p_left, p_right, p_top, p_bottom

    def compute_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes the output size of the convolution layer in advance.

        Args:
            input_size (Tuple[int, int]): Size of the input tensor (H, W).

        Returns:
            Tuple[int, int]: Size of the output tensor (H, W).
        """

        p_total_height = cast(int, self.padding[0])
        p_total_width = cast(int, self.padding[1])

        if self.is_padding_same:
            padding = self.internal_padding
            if not self.padding_precomputed:
                padding = self._compute_padding(input_size)

            p_total_width = padding[0] + padding[1]  # left + right
            p_total_height = padding[2] + padding[3]  # top + bottom

        # math.floor
        output_height = (
            input_size[0]
            + p_total_height
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        output_width = (
            input_size[1]
            + p_total_width
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        return output_height, output_width


class ConvTranspose2dExtended(nn.ConvTranspose2d):
    """
    Extended 2D transposed convolution layer mirroring TensorFlow with support for 'same' padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        """Initializes the ConvTranspose2dExtended layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolution kernel.
            stride (Union[int, Tuple[int, int]], optional): Stride of the convolution.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input.
                Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional): Dilation factor for the convolution.
                Defaults to 1.
            output_padding (Union[int, Tuple[int, int]], optional): Additional padding added to the output.
                Defaults to 0.
        """
        tmp_padding, is_padding_same = utils.standardize_padding(padding)

        # Convert attributes that have int values into tuples
        kernel_size_converted = utils.transform_int_to_tuple(kernel_size)
        stride_converted = utils.transform_int_to_tuple(stride)
        dilation_converted = utils.transform_int_to_tuple(dilation)
        output_padding_converted = utils.transform_int_to_tuple(output_padding)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size_converted,
            stride=stride_converted,
            padding=tmp_padding,
            dilation=dilation_converted,
            output_padding=output_padding_converted,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        if is_padding_same:
            self.padding, self.output_padding = self._compute_padding()

    # tensorflow assymetric padding
    def _compute_padding(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Computes the padding for the transposed convolution layer.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: Padding for height and width.
        """

        effective_kernel_height = utils.compute_effective_kernel_size(
            self.kernel_size[0], self.dilation[0]
        )
        effective_kernel_width = utils.compute_effective_kernel_size(
            self.kernel_size[1], self.dilation[1]
        )

        p_total_height = effective_kernel_height - self.stride[0]
        p_total_width = effective_kernel_width - self.stride[1]

        p_output_width = p_total_width % 2
        p_output_height = p_total_height % 2

        p_width = p_total_width // 2 + p_output_width
        p_height = p_total_height // 2 + p_output_height

        return (p_height, p_width), (p_output_height, p_output_width)

    def compute_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes the output size of the transposed convolution layer in advance.

        Args:
            input_size (Tuple[int, int]): Size of the input tensor (height, width).

        Returns:
            Tuple[int, int]: Size of the output tensor (height, width).
        """

        output_height = (
            (input_size[0] - 1) * cast(int, self.stride[0])
            - 2 * cast(int, self.padding[0])
            + self.dilation[0] * (self.kernel_size[0] - 1)
            + self.output_padding[0]
            + 1
        )
        output_width = (
            (input_size[1] - 1) * cast(int, self.stride[1])
            - 2 * cast(int, self.padding[1])
            + self.dilation[1] * (self.kernel_size[1] - 1)
            + self.output_padding[1]
            + 1
        )

        return output_height, output_width


class MaxPool2dExtended(nn.MaxPool2d):
    """Extended 2D max pooling layer mirroring TensorFlow with support for 'same' padding."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        *,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        input_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """Initializes the MaxPool2dExtended layer.

        Args:
            kernel_size (Union[int, Tuple[int, int]]): Size of the pooling kernel.
            stride (Union[int, Tuple[int, int]], optional): Stride of the pooling operation.
                Defaults to 1.
            padding (Union[int, Tuple[int, int]], optional): Padding added to both sides of the input.
                Defaults to 0.
            dilation (Union[int, Tuple[int, int]], optional): Dilation factor for the pooling operation.
                Defaults to 1.
            input_size (Optional[Tuple[int, int]], optional): Size of the input tensor (height, width).
                Defaults to None.
        """
        kernel_size = utils.transform_int_to_tuple(kernel_size)
        stride = utils.transform_int_to_tuple(stride)
        dilation = utils.transform_int_to_tuple(dilation)

        tmp_padding, is_padding_same = utils.standardize_padding(padding)
        self.is_padding_same = is_padding_same
        super().__init__(
            kernel_size, stride, padding=tmp_padding, dilation=dilation, **kwargs
        )

        # Ensuring that we work with tuples from now on
        self.kernel_size = cast(Tuple[int, int], self.kernel_size)
        self.stride = cast(Tuple[int, int], self.stride)
        self.dilation = cast(Tuple[int, int], self.dilation)

        self.padding_precomputed = True
        self.internal_padding: Tuple[int, int, int, int] = (0, 0, 0, 0)

        if is_padding_same:
            if input_size is None:
                self.padding_precomputed = False
            else:
                self.internal_padding = self._compute_padding(input_size)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass for the MaxPool2dExtended layer.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Output tensor of shape (N, C, H', W').
        """

        padded_input = self._pad_input(input)
        return super().forward(padded_input)  # type: ignore

    def _pad_input(self, x: Tensor) -> Tensor:
        """Pads the input tensor for the MaxPool2dExtended layer.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Padded input tensor.
        """

        if not self.is_padding_same:  # valid or number/tuple
            return x

        padding = self.internal_padding
        if not self.padding_precomputed:
            # (N, C, H, W)
            height, width = x.shape[-2], x.shape[-1]
            padding = self._compute_padding((height, width))

        x = F.pad(x, padding)

        return x

    def _compute_padding(
        self, input_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Computes the padding needed for the MaxPool2dExtended layer.

        Args:
            input_size (Tuple[int, int]): Size of the input tensor (height, width).

        Returns:
            Tuple[int, int, int, int]: Padding values (left, right, top, bottom).
        """

        # Type Checkers
        stride = cast(Tuple[int, int], self.stride)
        dilation = cast(Tuple[int, int], self.dilation)
        kernel_size = cast(Tuple[int, int], self.kernel_size)

        expected_output_height = (input_size[0] + stride[0] - 1) // stride[0]
        expected_output_width = (input_size[1] + stride[1] - 1) // stride[1]

        effective_kernel_height = utils.compute_effective_kernel_size(
            kernel_size[0], dilation[0]
        )
        effective_kernel_width = utils.compute_effective_kernel_size(
            kernel_size[1], dilation[1]
        )

        p_total_height = max(
            (expected_output_height - 1) * stride[0]
            + effective_kernel_height
            - input_size[0],
            0,
        )
        p_total_width = max(
            (expected_output_width - 1) * stride[1]
            + effective_kernel_width
            - input_size[1],
            0,
        )

        p_left, p_right = utils.split_padding(p_total_width)
        p_top, p_bottom = utils.split_padding(p_total_height)

        return p_left, p_right, p_top, p_bottom

    # should not be computing padding
    # if it was "valid" or something else (check also for other models)
    def compute_output_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes the output size for the MaxPool2dExtended layer in advance.

        Args:
            input_size (Tuple[int, int]): Size of the input tensor (H, W).

        Returns:
            Tuple[int, int]: Size of the output tensor (H', W).
        """

        p_total_height = self.padding[0]  # type: ignore
        p_total_width = self.padding[1]  # type: ignore

        kernel_size = cast(Tuple[int, int], self.kernel_size)
        stride = cast(Tuple[int, int], self.stride)
        dilation = cast(Tuple[int, int], self.dilation)

        if self.is_padding_same:
            padding = self.internal_padding
            if not self.padding_precomputed:
                padding = self._compute_padding(input_size)

            p_total_width = padding[0] + padding[1]  # 2*padding
            p_total_height = padding[2] + padding[3]  # 2*padding

        # math.floor
        output_height = (
            input_size[0] + p_total_height - dilation[0] * (kernel_size[0] - 1) - 1
        ) // stride[0] + 1
        output_width = (
            input_size[1] + p_total_width - dilation[1] * (kernel_size[1] - 1) - 1
        ) // stride[1] + 1

        return output_height, output_width
