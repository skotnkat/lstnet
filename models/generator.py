"""
Module implements the Generator model of LSTNet.
Generator is a LSTNet component responsible for generating data from a latent representation.
"""

from typing import Any, Dict, Tuple, Sequence
import torch.nn as nn

from models.lstnet_component import LstnetComponent
from models.extended_layers import ConvTranspose2dExtended


class Generator(LstnetComponent):
    """Generator model for LSTNet. Inherits from LstnetComponent."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels_num: int,
        params: Sequence[Dict[str, Any]],
        *,
        negative_slope: float = 0.01,
        momentum: float = 0.1,
        use_checkpoint: bool = False,
        **kwargs: Any,
    ):
        self.leaky_relu_negative_slope = negative_slope
        self.batch_norm_momentum = momentum

        # pass all the params apart the ones for the last layer
        super().__init__(
            input_size,
            in_channels_num,
            params,
            skip_last_layer=True,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )

        last_output_size = self.get_last_layer_output_size()
        last_out_channels_num = self.get_last_layer_out_channels()
        last_layer_idx = len(params) - 1

        _ = self._add_layer(
            last_output_size,
            last_out_channels_num,
            last_layer_idx,
            create_func=self._create_last_layer,
        )

    def _create_stand_layer(
        self, params: Dict[str, Any], in_channels: int, **kwargs: Dict[str, Any]
    ) -> nn.Sequential:
        layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params),
            nn.BatchNorm2d(params["out_channels"], momentum=self.batch_norm_momentum),
            nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope),
        )

        return layer

    @staticmethod
    def _create_last_layer(
        params: Dict[str, Any], in_channels: int, **kwargs: Any
    ) -> nn.Sequential:
        last_layer = nn.Sequential(
            ConvTranspose2dExtended(in_channels, **params), nn.Tanh()
        )

        # Mark as last layer for identification if needed
        setattr(last_layer, "_is_last_layer", True)

        return last_layer

    @staticmethod
    def _compute_layer_output_size(
        layer: nn.Sequential, input_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        return layer[0].compute_output_size(input_size)  # type: ignore (evaluate why not ok?)
