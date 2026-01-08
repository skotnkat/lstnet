"""
Module implements the Discriminator model of LSTNet.
Discriminator is a component responsible for distinguishing between real and generated data.
"""

from typing import Any, Dict, Tuple, Sequence
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.lstnet_component import LstnetComponent
from models.extended_layers import Conv2dExtended, MaxPool2dExtended


class Discriminator(LstnetComponent):
    """Discriminator model for LSTNet. Inherits from LstnetComponent."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels_num: int,
        params: Sequence[Any],
        *,
        negative_slope=0.01,
        use_checkpoint: bool = False,
        **kwargs
    ) -> None:
        """Initialize the Discriminator model.

        Args:
            input_size (Tuple[int, int]): Input size of the discriminator.
            in_channels_num (int): Number of input channels.
            params (Sequence[Any]): Parameters for convolutional and dense layers.
                The last element should contain parameters for the final dense layer.
            negative_slope (float, optional): Negative slope for LeakyReLU activation.
                Defaults to 0.01.
            use_checkpoint (bool, optional): Whether to use gradient checkpointing for
                memory efficiency during training. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self.dense_layer_params = dict(params[-1])
        self.leaky_relu_neg_slope = negative_slope

        # pass all the params apart from leaky relu and last layer
        super().__init__(
            input_size,
            in_channels_num,
            params[:-1],
            use_checkpoint=use_checkpoint,
            **kwargs
        )

        last_output_size = self.get_last_layer_output_size()
        last_layer_out_channels = self.get_last_layer_out_channels()

        in_features = (
            last_output_size[0] * last_output_size[1] * last_layer_out_channels  # type: ignore Does it return tuple? Why? What does it rerpesent?
        )

        self.dense_layer_params["in_features"] = in_features
        last_layer = self._create_last_layer()

        _ = self.layers.append(last_layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Discriminator model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self.layers, x, use_reentrant=False)
        else:
            return self.layers(x)  # returns raw

    def _create_stand_layer(
        self,
        params: Tuple[Dict[str, Any], Dict[str, Any]],
        in_channels: int,
        **kwargs: Dict[str, Any]
    ) -> nn.Sequential:
        """
        Create a standard layer for the Discriminator model.
        Standard layer consists of a convolutional layer
        followed by a LeakyReLU activation and a max pooling layer.

        Args:
            params (Tuple[Dict[str, Any], Dict[str, Any]]):
                Parameters for the convolutional and pooling layers.
            in_channels (int): Number of input channels.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If input size is not provided.

        Returns:
            nn.Sequential: Created standard layer.
        """
        conv_params, pool_params = params
        input_size_raw = kwargs.get("input_size", None)
        if (
            input_size_raw is None
            or not isinstance(input_size_raw, tuple)
            or len(input_size_raw) != 2
            or not all(isinstance(i, int) for i in input_size_raw)
        ):
            raise ValueError(
                "input_size must be provided for  Conv2dExtended layer to compute output size."
            )

        input_size: Tuple[int, int] = input_size_raw
        conv = Conv2dExtended(in_channels, input_size=input_size, **conv_params)
        relu = nn.LeakyReLU(negative_slope=self.leaky_relu_neg_slope)

        output_size = conv.compute_output_size(input_size)
        pool = MaxPool2dExtended(input_size=output_size, **pool_params)

        layer = nn.Sequential(conv, relu, pool)

        return layer

    def _create_last_layer(self) -> nn.Sequential:
        """Create the last layer for the Discriminator model.

        Returns:
            nn.Sequential: Last layer consisting of a flattening layer and a linear layer.
        """

        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(**self.dense_layer_params),
        )

        return last_layer

    @staticmethod
    def _compute_layer_output_size(
        layer: nn.Sequential, input_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        conv_output_size = layer[0].compute_output_size(input_size)

        pool_output_size = layer[2].compute_output_size(conv_output_size)

        return pool_output_size
