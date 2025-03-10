import json
import math
import numpy as np



PARAMS_FILE_PATH = "params.json"
NETWORK_NAMES = {"source_encoder", "target_encoder", "shared_encoder", "source_generator", "target_generator", "shared_generator", \
                 "source_discriminator", "target_discriminator", "latent_discriminator"}

LAYERS_NUM = "stand_layers_num"
OUT_CHANNELS_NUM = "out_channels"
KERNEL_SIZE = "kernel_sizes"
STRIDE = "strides"
PADDING = "paddings"
POOLING_KERNEL_SIZE = "pooling_kernels"
POOLING_STRIDE = "pooling_strides"


def get_networks_params():
    with open(PARAMS_FILE_PATH, "r") as file:
        params = json.load(file)
    
    missing_network_params = NETWORK_NAMES.difference(set(params.keys()))
    if len(missing_network_params):
        raise ValueError(f"Missing Input Parameters for Network Parts: {missing_network_params}")
    
    return params


def initialize_weights():
    return 20, 20, 100, 100, 100, 100, 30


def check_and_transform_params(params, expected_params):  # params is implicitly being changed
    for param in expected_params:
        if param not in params:
            raise ValueError(f"Parameter {param} is not specified")
    
    layers_num = params[LAYERS_NUM]
    for param in expected_params:
        if param == LAYERS_NUM:
            continue
    
        values = params[param]
        if type(values) != list:
            params[param] = [values] * layers_num

        elif len(values) != layers_num:  # is of type list
            raise ValueError(F"Paramter {param} has not enough values specified ({len(values)} specified but network has {num_layers} layers.")



def obtain_last_value(param):
    if type(param) != list:
        return param

    return param[-1]





