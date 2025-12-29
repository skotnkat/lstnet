"""
Module implements a base class (nn.Module) for LSTNet components.
From which other classes like Encoder, Generator and Discriminator inherit.
"""

from typing import Any, Callable, Dict, List, Tuple, Sequence
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from torch import Tensor


class LstnetComponent(ABC, nn.Module):
    """Base class for LSTNET components."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels_num: int,
        params: Sequence[Any],
        *,
        skip_last_layer: bool = False,
        use_checkpoint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()  # type: ignore

        self.layers: nn.Sequential = nn.Sequential()
        self.output_sizes: List[Tuple[int, int]] = []
        self.params = params
        self.use_checkpoint = use_checkpoint

        stand_layers_num: int = len(self.params) - skip_last_layer
        for i in range(stand_layers_num):
            # passing only convolution
            output_size = self._add_layer(
                input_size, in_channels_num, i, self._create_stand_layer
            )

            input_size = output_size
            in_channels_num = self.get_last_layer_out_channels()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self.layers, x, use_reentrant=False)
        else:
            return self.layers(x)

    def _add_layer(
        self,
        input_size: Tuple[int, int],
        in_channels_num: int,
        idx: int,
        create_func: Callable,
    ) -> Tuple[int, int]:
        """Add a layer to the model.

        Args:
            input_size (Tuple[int, int]): Input size of the layer.
            in_channels_num (int): Number of input channels.
            idx (int): Index of the layer.
            create_func (Callable): Function to create the layer. Is specified by the child class.

        Returns:
            Tuple[int, int]: Output size of the layer.
        """

        layer = create_func(self.params[idx], in_channels_num, input_size=input_size)
        _ = self.layers.append(layer)

        output_size = self._compute_layer_output_size(layer, input_size)
        self.output_sizes.append(output_size)

        return output_size

    @abstractmethod
    def _create_stand_layer(
        self, params: Any, in_channels: int, **kwargs: Dict[str, Any]
    ) -> nn.Sequential:
        """Abstract function. Create a standard layer.

        Args:
            params (Any): Parameters for the layer.
            in_channels (int): Number of input channels.
            **kwargs: Additional keyword arguments.

        Returns:
            nn.Sequential: Created standard layer.
        """

    @staticmethod
    @abstractmethod
    def _compute_layer_output_size(
        layer: nn.Sequential, input_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Abstract function. Compute the output size of a layer.

        Args:
            layer (nn.Sequential): Layer to compute output size for.
            input_size (Tuple[int, int]): Size of the input to be passed through the layer.

        Returns:
            Tuple[int, int]: Size of the output from the layer.
        """

    def get_last_layer_out_channels(self) -> int:
        """Get the number of output channels from the last layer."""
        out_channels: int = self.layers[-1][0].out_channels  # type: ignore (check what is happening: is it nn.Module or nn.Sequential)
        return out_channels

    def get_last_layer_output_size(self) -> Tuple[int, int]:
        """Get the output size of the last layer."""
        return self.output_sizes[-1]
