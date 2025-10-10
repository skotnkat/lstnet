"""
Module implements the Encoder model of LSTNet.
Encoder is a LSTNet component responsible for encoding input data into a latent representation.
"""

from typing import Any, Dict, Tuple, Sequence
import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import Conv2dExtended


class Encoder(LstnetComponent):
    """Encoder model for LSTNet. Inherits from LstnetComponent."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels_num: int,
        params: Sequence[Dict[str, Any]],
        *,
        negative_slope: float = 0.01,
        momentum: float = 0.1,
        **kwargs: Any,
    ):
        self.leaky_relu_negative_slope = negative_slope
        self.batch_norm_momentum = momentum

        super().__init__(input_size, in_channels_num, params, **kwargs)

    def _create_stand_layer(
        self, params: Dict[str, Any], in_channels: int, **kwargs: Dict[str, Any]
    ) -> nn.Sequential:
        """
        Create a standard layer for the Encoder model.
        Standard layer consists of a convolutional layer followed by
        a BatchNorm layer and a LeakyReLU activation.

        Args:
            params (Dict[str, Any]): Parameters for the convolutional layer.
            in_channels (int): Number of input channels.
            **kwargs: Additional keyword arguments.

        Returns:
            nn.Sequential: Created standard layer.
        """
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
        layer = nn.Sequential(
            Conv2dExtended(in_channels, input_size=input_size, **params),
            nn.BatchNorm2d(params["out_channels"], momentum=self.batch_norm_momentum),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
        )

        return layer

    @staticmethod
    def _compute_layer_output_size(
        layer: nn.Sequential, input_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        return layer[0].compute_output_size(input_size)  # type: ignore (why not ok?)
