import json


PARAMS_FILE_PATH = None
NETWORK_NAMES = {"first_encoder", "second_encoder", "shared_encoder", "first_generator", "second_generator",
                 "shared_generator", "first_discriminator", "second_discriminator", "latent_discriminator"}

OUTPUT_FOLDER = None
LOSS_FILE = None
BATCH_SIZE = None
ADAM_LR = None
ADAM_DECAY = None
DELTA_LOSS = None

DEVICE = None


def get_networks_params():
    with open(PARAMS_FILE_PATH, "r") as file:
        params = json.load(file)
    
    missing_network_params = NETWORK_NAMES.difference(set(params.keys()))
    if len(missing_network_params):
        raise ValueError(f"Missing Input Parameters for Network Parts: {missing_network_params}")
    
    return params


def initialize_weights():
    return 20, 20, 100, 100, 100, 100, 30


def is_padding_needed(params):
    padding = params["padding"]
    del params["padding"]

    to_pad = (padding == "same")

    return to_pad


def transform_int_to_tuple(x):
    if isinstance(x, int):
        x = (x, x)

    return x


def standardize_padding(padding):
    padding = transform_int_to_tuple(padding)
    is_padding_same = (padding == "same")
    
    if not isinstance(padding, tuple):
        padding = (0, 0)

    return padding, is_padding_same


def split_padding(p_total):
    p_first = p_total // 2
    p_second = p_total - p_first

    return p_first, p_second


def compute_effective_kernel_size(kernel_size, dilation):
    return dilation * (kernel_size - 1) + 1
