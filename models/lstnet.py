from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
import utils

import torch.nn as nn

FIRST_INPUT_SHAPE = None
SECOND_INPUT_SHAPE = None

FIRST_IN_CHANNELS_NUM = None
SECOND_IN_CHANNELS_NUM = None


class LSTNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_encoder = None
        self.second_encoder = None
        self.shared_encoder = None
        
        self.first_generator = None
        self.second_generator = None
        self.shared_generator = None
        
        self.first_discriminator = None
        self.second_discriminator = None
        self.latent_discriminator = None

        if (FIRST_INPUT_SHAPE is None) or (FIRST_IN_CHANNELS_NUM is None) \
                or (SECOND_INPUT_SHAPE is None) or (SECOND_IN_CHANNELS_NUM is None):
            raise ValueError("Missing one of the required global variables: FIRST_INPUT_SHAPE, FIRST_IN_CHANNELS_NUM, \
                                SECOND_INPUT_SHAPE, SECOND_IN_CHANNELS_NUM")

        params = utils.get_networks_params()

        self.initialize_encoders(FIRST_INPUT_SHAPE, SECOND_INPUT_SHAPE,
                                 FIRST_IN_CHANNELS_NUM, SECOND_IN_CHANNELS_NUM,
                                 params)

        self.initialize_generators(params)

        self.initialize_discriminators(FIRST_INPUT_SHAPE, SECOND_INPUT_SHAPE,
                                       FIRST_IN_CHANNELS_NUM, SECOND_IN_CHANNELS_NUM,
                                       params)

    def initialize_encoders(self, first_input_size, second_input_size,
                            first_in_channels_num, second__in_channels_num, params):
        self.first_encoder = Encoder(first_input_size, first_in_channels_num, params["first_encoder"])
        self.second_encoder = Encoder(second_input_size, second__in_channels_num, params["second_encoder"])

        input_size_shared = self.first_encoder.get_last_layer_output_size()
        in_channels_num_shared = self.first_encoder.get_last_layer_out_channels()
        self.shared_encoder = Encoder(input_size_shared, in_channels_num_shared, params=params["shared_encoder"])

    def initialize_generators(self, params):
        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()
        
        self.shared_generator = Generator(input_size_shared, out_channels_shared, params["shared_generator"])

        input_size = self.shared_generator.get_last_layer_output_size()
        
        out_channels = self.shared_generator.get_last_layer_out_channels()

        self.first_generator = Generator(input_size, out_channels, params["first_generator"])
        self.second_generator = Generator(input_size, out_channels, params["second_generator"])

    def initialize_discriminators(self, first_input_size, second_input_size,
                                  first_in_channels_num, second_in_channels_num, params):
        self.first_discriminator = Discriminator(first_input_size, first_in_channels_num, params["first_discriminator"])
        self.second_discriminator = Discriminator(second_input_size, second_in_channels_num, params["second_discriminator"])

        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()
        self.latent_discriminator = Discriminator(input_size_shared, out_channels_shared, params["latent_discriminator"])

    def map_first_to_latent(self, x_first):
        x_latent = self.first_encoder.forward(x_first)
        return self.shared_encoder.forward(x_latent)

    def map_second_to_latent(self, x_second):
        x_latent = self.second_encoder.forward(x_second)
        return self.shared_encoder.forward(x_latent)

    def map_latent_to_first(self, x_latent):
        x_first = self.shared_generator.forward(x_latent)
        return self.first_generator.forward(x_first)
    
    def map_latent_to_second(self, x_latent):
        x_second = self.shared_generator.forward(x_latent)
        return self.second_generator.forward(x_second)

    def map_first_to_second(self, x_first, return_latent=False):
        x_latent = self.map_first_to_latent(x_first)
        x_second = self.map_latent_to_second(x_latent)

        if return_latent:
            return x_second, x_latent

        return x_second

    def map_second_to_first(self, x_second, return_latent=False):
        x_latent = self.map_second_to_latent(x_second)
        x_first = self.map_latent_to_first(x_latent)

        if return_latent:
            return x_first, x_latent

        return x_first
