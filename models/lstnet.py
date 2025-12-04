"""LSTNET model class for img-to-img translation (supervised, unsupervised)."""

from typing import Tuple, Dict, Any, Union, overload, Literal
import copy

# from utils import ModeCollapseDetected
from torch import Tensor
import torch
import torch.nn as nn


from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator

import utils


# TO DO: COmpile Config (mode, dynamic, fullgraph, etc.)


class LSTNET(nn.Module):
    """LSTNET model for image-to-image translation."""

    def __init__(
        self,
        first_domain_name: str,
        second_domain_name: str,
        params: Dict[str, Any],
        *,
        first_input_size: Tuple[int, int],
        second_input_size: Tuple[int, int],
        first_in_channels_num: int = 1,
        second_in_channels_num: int = 1,
        compile_components: bool = False,
    ) -> None:
        """Initialize the LSTNET model.

        Args:
            first_domain_name (str): Name of the first domain.
            second_domain_name (str): Name of the second domain.
            params (Dict[str, Any]): Model parameters.
            first_input_size (Tuple[int, int]): Input size for the first domain.
            second_input_size (Tuple[int, int]): Input size for the second domain.
            first_in_channels_num (int, optional): Number of input channels for the first domain.
                Defaults to 1.
            second_in_channels_num (int, optional): Number of input channels for the second domain.
                Defaults to 1.
        """
        super().__init__()

        params = copy.deepcopy(params)

        # Across multiple lstnet components
        self.global_params = {**params["leaky_relu"], **params["batch_norm"]}
        del params["leaky_relu"]
        del params["batch_norm"]

        self.first_domain_name = first_domain_name
        self.second_domain_name = second_domain_name

        self.first_input_size = first_input_size
        self.second_input_size = second_input_size
        self.first_in_channels_num = first_in_channels_num
        self.second_in_channels_num = second_in_channels_num

        self.params = copy.deepcopy(params)

        # Somehow mention that it has other self. attributes
        self.initialize_encoders()
        self.initialize_generators()
        self.initialize_discriminators()

        self.disc_params = (
            list(self.first_discriminator.parameters())
            + list(self.second_discriminator.parameters())
            + list(self.latent_discriminator.parameters())
        )

        self.enc_gen_params = (
            list(self.first_encoder.parameters())
            + list(self.second_encoder.parameters())
            + list(self.shared_encoder.parameters())
            + list(self.first_generator.parameters())
            + list(self.second_generator.parameters())
            + list(self.shared_generator.parameters())
        )

        if compile_components:
            self.compile_components()

        print("LSTNET model initialized")

    def compile_components(
        self, dynamic: bool = False, mode: str = "max-autotune"
    ) -> None:
        print("Compiling LSTNET components with torch.compile()...")
        self.first_encoder = torch.compile(
            self.first_encoder, dynamic=dynamic, mode=mode
        )
        self.second_encoder = torch.compile(
            self.second_encoder, dynamic=dynamic, mode=mode
        )
        self.shared_encoder = torch.compile(
            self.shared_encoder, dynamic=dynamic, mode=mode
        )

        self.first_generator = torch.compile(
            self.first_generator, dynamic=dynamic, mode=mode
        )
        self.second_generator = torch.compile(
            self.second_generator, dynamic=dynamic, mode=mode
        )
        self.shared_generator = torch.compile(
            self.shared_generator, dynamic=dynamic, mode=mode
        )
        self.first_discriminator = torch.compile(
            self.first_discriminator, dynamic=dynamic, mode=mode
        )
        self.second_discriminator = torch.compile(
            self.second_discriminator, dynamic=dynamic, mode=mode
        )
        self.latent_discriminator = torch.compile(
            self.latent_discriminator, dynamic=dynamic, mode=mode
        )

    def initialize_encoders(self) -> None:
        """
        Initialize the encoders (first, second, shared).
        First encoder processes images from first domain.
        Second encoder processes images from second domain.
        Shared encoder processes the output of the first and second encoders
            and maps them to a common latent space.

        Raises:
            ValueError: If the output sizes or out channels
                of the first and second encoders do not match.
        """

        self.first_encoder = Encoder(
            self.first_input_size,
            self.first_in_channels_num,
            self.params["first_encoder"],
            **self.global_params,
        )
        self.second_encoder = Encoder(
            self.second_input_size,
            self.second_in_channels_num,
            self.params["second_encoder"],
            **self.global_params,
        )

        first_input_size = self.first_encoder.get_last_layer_output_size()
        first_in_channels_num = self.first_encoder.get_last_layer_out_channels()

        second_input_size = self.second_encoder.get_last_layer_output_size()
        second_in_channels_num = self.second_encoder.get_last_layer_out_channels()

        # Check the compatibility of the encoders' output sizes and out channels
        if (first_input_size != second_input_size) or (
            first_in_channels_num != second_in_channels_num
        ):
            raise ValueError(
                "The output sizes or out channels of the first and second encoders do not match."
            )

        # Shared encoder input size and in channels
        self.shared_encoder = Encoder(
            first_input_size,
            first_in_channels_num,
            params=self.params["shared_encoder"],
            **self.global_params,
        )

    def initialize_generators(self) -> None:
        """
        Initialize the generators (first, second, shared).
        Shared generator processes the latent representation
            and maps it to an intermediate representation.
        First generator processes the output of the shared generator
            and maps it to the first domain.
        Second generator processes the output of the shared generator
            and maps it to the second domain.

        Raises:
            ValueError:
                If the output sizes or out channels of the generators
                do not match the input sizes or in channels of the encoders.
        """
        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()

        self.shared_generator = Generator(
            input_size_shared,
            out_channels_shared,
            self.params["shared_generator"],
            **self.global_params,
        )

        input_size = self.shared_generator.get_last_layer_output_size()
        out_channels = self.shared_generator.get_last_layer_out_channels()

        self.first_generator = Generator(
            input_size,
            out_channels,
            self.params["first_generator"],
            **self.global_params,
        )
        self.second_generator = Generator(
            input_size,
            out_channels,
            self.params["second_generator"],
            **self.global_params,
        )

        first_gen_output_size = self.first_generator.get_last_layer_output_size()
        first_gen_out_channels = self.first_generator.get_last_layer_out_channels()
        if (first_gen_output_size != self.first_input_size) or (
            first_gen_out_channels != self.first_in_channels_num
        ):
            raise ValueError(
                f"Size mismatch for {self.first_domain_name}: \
                    The output size or out channels of the first generator do not match\
                        the input size or in channels of the first encoder."
            )

        second_gen_output_size = self.second_generator.get_last_layer_output_size()
        second_gen_out_channels = self.second_generator.get_last_layer_out_channels()
        if (second_gen_output_size != self.second_input_size) or (
            second_gen_out_channels != self.second_in_channels_num
        ):
            raise ValueError(
                f"Size mismatch for {self.second_domain_name}: \
                    The output size or out channels of the second generator do not match\
                        the input size or in channels of the second encoder."
            )

    def initialize_discriminators(self) -> None:
        """
        Initialize the discriminators (first, second, latent) that distinguish
            between real and generated images in the respective domains.
        Discriminators are used to train the generators to produce realistic images.
        Real images should be classified as 1 and generated (fake) images as 0.

        First discriminator distinguishes between real and generated images in the first domain.
        Second discriminator distinguishes between real and generated images in the second domain.
        Latent discriminator distinguishes between
            latent representations of real images from both domains.
        """
        self.first_discriminator = Discriminator(
            self.first_input_size,
            self.first_in_channels_num,
            self.params["first_discriminator"],
            **self.global_params,
        )
        self.second_discriminator = Discriminator(
            self.second_input_size,
            self.second_in_channels_num,
            self.params["second_discriminator"],
            **self.global_params,
        )

        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()
        self.latent_discriminator = Discriminator(
            input_size_shared,
            out_channels_shared,
            self.params["latent_discriminator"],
            **self.global_params,
        )

    def map_first_to_latent(self, x_first: Tensor) -> Tensor:
        """Maps images from the first domain to the latent space.

        Args:
            x_first (Tensor): Images from the first domain.

        Returns:
            Tensor: Latent representation of the input images.
        """

        x_latent = self.first_encoder(x_first)
        return self.shared_encoder(x_latent)

    def map_second_to_latent(self, x_second: Tensor) -> Tensor:
        """Maps images from the second domain to the latent space.

        Args:
            x_second (Tensor): Images from the second domain.

        Returns:
            Tensor: Latent representation of the input images.
        """

        x_latent = self.second_encoder(x_second)
        return self.shared_encoder(x_latent)

    def map_latent_to_first(self, x_latent: Tensor) -> Tensor:
        """Maps latent representations to the first domain.

        Args:
            x_latent (Tensor): Latent representations.

        Returns:
            Tensor: Generated images in the first domain.
        """

        x_first = self.shared_generator(x_latent)
        return self.first_generator(x_first)

    def map_latent_to_second(self, x_latent: Tensor) -> Tensor:
        """Maps latent representations to the second domain.

        Args:
            x_latent (Tensor): Latent representations.

        Returns:
            Tensor: Generated images in the second domain.
        """

        x_second = self.shared_generator(x_latent)
        return self.second_generator(x_second)

    def map_first_to_second(self, x_first: Tensor) -> Tuple[Tensor, Tensor]:
        """Maps images from the first domain to the second domain.

        Args:
            x_first (Tensor): Images from the first domain.

        Returns:
            Tuple[Tensor, Tensor]: Generated images in the second domain
                or a tuple of generated images and latent representation.
        """
        x_latent = self.map_first_to_latent(x_first)
        x_second = self.map_latent_to_second(x_latent)

        return x_second, x_latent

    def map_second_to_first(self, x_second: Tensor) -> Tuple[Tensor, Tensor]:
        """Maps images from the second domain to the first domain.

        Args:
            x_second (Tensor): Images from the second domain.

        Returns:
            Tuple[Tensor, Tensor]: Generated images in the first domain and latent representation.
        """
        x_latent = self.map_second_to_latent(x_second)
        x_first = self.map_latent_to_first(x_latent)

        return x_first, x_latent

    def get_cc_components(
        self,
        first_gen: Tensor,
        second_gen: Tensor,
        first_latent: Tensor,
        second_latent: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Gets the cycle-consistent components from the generated images and latent representations.

        Args:
            first_gen (Tensor): Generated images in the first domain.
            second_gen (Tensor): Generated images in the second domain.
            first_latent (Tensor): Latent representation of images from the first domain.
            second_latent (Tensor): Latent representation of images from the second domain.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Cycle-consistent components.
        """
        # map latent representation of real first images back to first domain
        first_cycle = self.map_latent_to_first(first_latent)

        # map latent representation of real second images back to second domain
        second_cycle = self.map_latent_to_second(second_latent)

        # map generated images in second domain back to first domain
        first_full_cycle, _ = self.map_second_to_first(second_gen)

        # map generated images in first domain back to second domain
        second_full_cycle, _ = self.map_first_to_second(first_gen)

        return first_cycle, second_cycle, first_full_cycle, second_full_cycle

    def forward(
        self,
        x: Tensor,
        map_first_to_second=True,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Maps images between two domains.

        Args:
            x (Tensor): Input images.
            map_first_to_second (bool, optional):
                Whether to map from the first domain to the second or vice versa.
                If True, maps from first to second. If False, maps from second to first.
                Defaults to True.

        Returns:
            Tuple[Tensor, Tensor]: Generated images and latent representation.
        """

        if map_first_to_second:
            return self.map_first_to_second(x)

        return self.map_second_to_first(x)

    def set_domain_name(
        self, name: str, first: bool = True
    ) -> None:  # Is it called somewhere? If not, remove it.
        """Sets the domain name for the model.

        Args:
            name (str): The name of the domain.
            first (bool, optional): Whether to set the name for the first domain. Defaults to True.
        """
        if first:
            self.first_domain_name = name

        else:
            self.second_domain_name = name

    def get_lstnet_state_dict(self) -> Dict[str, Any]:
        self.params["leaky_relu"] = {
            "negative_slope": self.global_params["negative_slope"]
        }

        self.params["batch_norm"] = {"momentum": self.global_params["momentum"]}

        attr_dict = {
            "domain_name": [self.first_domain_name, self.second_domain_name],
            "input_sizes": [self.first_input_size, self.second_input_size],
            "in_channels_num": [
                self.first_in_channels_num,
                self.second_in_channels_num,
            ],
            "params": self.params,
        }

        dict_to_save = {
            "attr_dict": attr_dict,
            "state_dict": self.state_dict(),
        }

        return dict_to_save

    def save_model(self, output_path: str) -> None:
        """Saves the model to the specified output path.

        Args:
            output_path (str): The path to save the model.
        """

        dict_to_save = self.get_lstnet_state_dict()

        torch.save(dict_to_save, output_path)

    @staticmethod
    def load_lstnet_from_state_dict(state_dict: Dict[str, Any]) -> "LSTNET":
        """Loads the LSTNET model from the specified state dictionary.

        Args:
            state_dict (Dict[str, Any]): The state dictionary containing model attributes and state.

        Returns:
            LSTNET: The loaded LSTNET model.
        """
        attr_dict = state_dict["attr_dict"]
        state_dict = state_dict["state_dict"]

        first_domain_name, second_domain_name = attr_dict["domain_name"]
        first_input_size, second_input_size = attr_dict["input_sizes"]
        first_in_channels_num, second_in_channels_num = attr_dict["in_channels_num"]

        params = attr_dict["params"]

        model = LSTNET(
            first_domain_name,
            second_domain_name,
            params,
            first_input_size=first_input_size,
            second_input_size=second_input_size,
            first_in_channels_num=first_in_channels_num,
            second_in_channels_num=second_in_channels_num,
        )

        load_result = model.load_state_dict(state_dict)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"Missing keys: {load_result.missing_keys}")
            print(f"Unexpected keys: {load_result.unexpected_keys}")

        return model

    @staticmethod
    def load_lstnet_model(input_path: str) -> "LSTNET":
        """Loads the LSTNET model from the specified input path.

        Args:
            input_path (str): The path to the model file.

        Returns:
            LSTNET: The loaded LSTNET model.
        """
        dict_to_load = torch.load(input_path, map_location=utils.DEVICE)
        return LSTNET.load_lstnet_from_state_dict(dict_to_load)
