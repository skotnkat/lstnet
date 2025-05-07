from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
import utils

import torch.nn as nn
import torch
from torch.optim import Adam
import functools
import operator
import torch.nn.init as init

import loss_functions


class LSTNET(nn.Module):
    def __init__(self, first_domain_name="", second_domain_name="", params=None):
        super().__init__()

        self.first_domain_name = first_domain_name
        self.second_domain_name = second_domain_name

        self.first_encoder = None
        self.second_encoder = None
        self.shared_encoder = None

        self.first_generator = None
        self.second_generator = None
        self.shared_generator = None

        self.first_discriminator = None
        self.second_discriminator = None
        self.latent_discriminator = None

        if (utils.FIRST_INPUT_SHAPE is None) or (utils.FIRST_IN_CHANNELS_NUM is None) \
                or (utils.SECOND_INPUT_SHAPE is None) or (utils.SECOND_IN_CHANNELS_NUM is None):
            raise ValueError(
                "Missing one of the required global variables: FIRST_INPUT_SHAPE, FIRST_IN_CHANNELS_NUM, SECOND_INPUT_SHAPE, SECOND_IN_CHANNELS_NUM")

        self.first_input_size = utils.FIRST_INPUT_SHAPE
        self.second_input_size = utils.SECOND_INPUT_SHAPE
        self.first_in_channels_num = utils.FIRST_IN_CHANNELS_NUM
        self.second_in_channels_num = utils.SECOND_IN_CHANNELS_NUM

        self.params = params
        if self.params is None:
            self.params = utils.get_networks_params()

        self.initialize_encoders()

        self.initialize_generators()

        self.initialize_discriminators()

        self.disc_params = list(self.first_discriminator.parameters()) \
                           + list(self.second_discriminator.parameters()) \
                           + list(self.latent_discriminator.parameters())

        self.enc_gen_params = list(self.first_encoder.parameters()) \
                              + list(self.second_encoder.parameters()) \
                              + list(self.shared_encoder.parameters()) \
                              + list(self.first_generator.parameters()) \
                              + list(self.second_generator.parameters()) \
                              + list(self.shared_generator.parameters())

        print(f'Adam prameters - learning rate: {utils.ADAM_LR}, betas: {utils.ADAM_DECAY}')
        if utils.ADAM_LR is not None and utils.ADAM_DECAY is not None:
            self.disc_optim = Adam(self.disc_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY, amsgrad=True)
            self.enc_gen_optim = Adam(self.enc_gen_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY, amsgrad=True)

        # print('Setting init distribution as glorot uniform')
        # self.apply(custom_init)

        print('LSTNET model initialized')


    def initialize_encoders(self):
        self.first_encoder = Encoder(self.first_input_size, self.first_in_channels_num, self.params["first_encoder"])
        self.second_encoder = Encoder(self.second_input_size, self.second_in_channels_num, self.params["second_encoder"])

        input_size_shared = self.first_encoder.get_last_layer_output_size()
        in_channels_num_shared = self.first_encoder.get_last_layer_out_channels()
        self.shared_encoder = Encoder(input_size_shared, in_channels_num_shared, params=self.params["shared_encoder"])

    def initialize_generators(self):
        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()

        self.shared_generator = Generator(input_size_shared, out_channels_shared, self.params["shared_generator"])

        input_size = self.shared_generator.get_last_layer_output_size()

        out_channels = self.shared_generator.get_last_layer_out_channels()

        self.first_generator = Generator(input_size, out_channels, self.params["first_generator"])
        self.second_generator = Generator(input_size, out_channels, self.params["second_generator"])

    def initialize_discriminators(self):
        self.first_discriminator = Discriminator(self.first_input_size, self.first_in_channels_num, self.params["first_discriminator"])
        self.second_discriminator = Discriminator(self.second_input_size, self.second_in_channels_num,
                                                  self.params["second_discriminator"])

        input_size_shared = self.shared_encoder.get_last_layer_output_size()
        out_channels_shared = self.shared_encoder.get_last_layer_out_channels()
        self.latent_discriminator = Discriminator(input_size_shared, out_channels_shared,
                                                  self.params["latent_discriminator"])

    def map_first_to_latent(self, x_first):
        x_latent = self.first_encoder.forward(x_first)
        return self.shared_encoder.forward(x_latent).to(utils.DEVICE)

    def map_second_to_latent(self, x_second):
        x_latent = self.second_encoder.forward(x_second)
        return self.shared_encoder.forward(x_latent).to(utils.DEVICE)

    def map_latent_to_first(self, x_latent):
        x_first = self.shared_generator.forward(x_latent)
        return self.first_generator.forward(x_first).to(utils.DEVICE)

    def map_latent_to_second(self, x_latent):
        x_second = self.shared_generator.forward(x_latent)
        return self.second_generator.forward(x_second).to(utils.DEVICE)

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

    def get_cc_components(self, first_gen, second_gen, first_latent, second_latent):
        # map latent representation of real first images back to first domain
        first_cycle = self.map_latent_to_first(first_latent)

        # map latent representation of real second images back to second domain
        second_cycle = self.map_latent_to_second(second_latent)

        # map generated images in second domain back to first domain
        first_full_cycle = self.map_second_to_first(second_gen)

        # map generated images in first domain back to second domain
        second_full_cycle = self.map_first_to_second(first_gen)

        return first_cycle, second_cycle, first_full_cycle, second_full_cycle

    def set_domain_name(self, name, first=True):
        if first:
            self.first_domain_name = name

        else:
            self.second_domain_name = name

    def save_model(self, output_path):
        attr_dict = {
            'domain_name': [self.first_domain_name, self.second_domain_name],
            'input_shape': [self.first_input_size, self.second_input_size],
            'in_channels_num': [self.first_in_channels_num, self.second_in_channels_num],
            'params' : self.params
        }

        dict_to_save = {
            'attr_dict': attr_dict,
            'state_dict': self.state_dict()
        }

        torch.save(dict_to_save, output_path)

    def run_networks(self, first_real, second_real):
        second_gen, first_latent = self.map_first_to_second(first_real, return_latent=True)
        first_gen, second_latent = self.map_second_to_first(second_real, return_latent=True)

        return first_gen, second_gen, first_latent, second_latent

    @staticmethod
    def load_lstnet_model(input_path):
        dict_to_load = torch.load(input_path, map_location=utils.DEVICE)
        attr_dict = dict_to_load['attr_dict']
        state_dict = dict_to_load['state_dict']

        first_domain_name, second_domain_name = attr_dict['domain_name']
        utils.FIRST_INPUT_SHAPE, utils.SECOND_INPUT_SHAPE = attr_dict['input_shape']
        utils.FIRST_IN_CHANNELS_NUM, utils.SECOND_IN_CHANNELS_NUM = attr_dict['in_channels_num']

        model = LSTNET(first_domain_name, second_domain_name, params=attr_dict['params'])
        model.load_state_dict(state_dict)

        return model

    def update_disc(self, first_real, second_real):
        self.disc_optim.zero_grad()

        with torch.no_grad():
            imgs_mapping = self.run_networks(first_real, second_real)  # generated images and latent

        disc_loss_tuple = loss_functions.compute_discriminator_loss(self, first_real, second_real, *imgs_mapping)

        total_disc_loss = functools.reduce(operator.add, disc_loss_tuple)
        total_disc_loss.backward()

        self.disc_optim.step()

    def update_enc_gen(self, first_real, second_real):
        self.enc_gen_optim.zero_grad()

        imgs_mapping = self.run_networks(first_real, second_real)  # generated images and latent
        imgs_cc = self.get_cc_components(*imgs_mapping)

        cc_loss_tuple = loss_functions.compute_cc_loss(first_real, second_real, *imgs_cc)
        enc_gen_loss_tuple = loss_functions.compute_enc_gen_loss(self, *imgs_mapping)

        total_enc_gen_loss = functools.reduce(operator.add, cc_loss_tuple) + functools.reduce(operator.add,
                                                                                              enc_gen_loss_tuple)
        total_enc_gen_loss.backward()
        self.enc_gen_optim.step()

    def run_eval_loop(self, first_real, second_real):
        with torch.no_grad():
            imgs_mapping = self.run_networks(first_real, second_real)
            imgs_cc = self.get_cc_components(*imgs_mapping)

            disc_loss_tuple = loss_functions.compute_discriminator_loss(self, first_real, second_real, *imgs_mapping, return_grad=False)
            cc_loss_tuple = loss_functions.compute_cc_loss(first_real, second_real, *imgs_cc, return_grad=False)

        return disc_loss_tuple, cc_loss_tuple


def custom_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(m, '_is_last_layer') and m._is_last_layer:
            init.xavier_uniform_(m.weight)
        else:
            init.kaiming_normal_(m.weight, a=0.3, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)


def init_kaiming_normal_leaky_relu(m):
    """
    Custom initialization function to apply Kaiming Normal
    with the correct negative slope for Leaky ReLU (a=0.3).

    Args:
        m (nn.Module): The module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.kaiming_normal_(m.weight, a=0.3, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)