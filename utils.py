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
NUM_WORKERS = None

FIRST_INPUT_SHAPE, SECOND_INPUT_SHAPE = None, None
FIRST_IN_CHANNELS_NUM, SECOND_IN_CHANNELS_NUM = None, None

DISC_LOSSES = {'first_loss': [], 'second_loss': [], 'latent_loss': []}
CC_LOSSES = {'first_cycle': [], 'second_cycle': [], 'first_full_cycle': [], 'second_full_cycle': []}
ENC_GEN_LOSSES = {'first_loss': [], 'second_loss': [], 'latent_loss': []}


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


def init_epoch_loss():
    DISC_LOSSES['first_loss'].append({'train': 0, 'val': 0})
    DISC_LOSSES['second_loss'].append({'train': 0, 'val': 0})
    DISC_LOSSES['latent_loss'].append({'train': 0, 'val': 0})

    CC_LOSSES['first_cycle'].append({'train': 0, 'val': 0})
    CC_LOSSES['second_cycle'].append({'train': 0, 'val': 0})
    CC_LOSSES['first_full_cycle'].append({'train': 0, 'val': 0})
    CC_LOSSES['second_full_cycle'].append({'train': 0, 'val': 0})

    ENC_GEN_LOSSES['first_loss'].append({'train': 0, 'val': 0})
    ENC_GEN_LOSSES['second_loss'].append({'train': 0, 'val': 0})
    ENC_GEN_LOSSES['latent_loss'].append({'train': 0, 'val': 0})


def log_epoch_loss(disc_loss, enc_gen_loss, cc_loss, cur_epoch, op='train'):
    DISC_LOSSES['first_loss'][cur_epoch][op] += disc_loss[0].item()
    DISC_LOSSES['second_loss'][cur_epoch][op] += disc_loss[1].item()
    DISC_LOSSES['latent_loss'][cur_epoch][op] += disc_loss[2].item()

    CC_LOSSES['first_cycle'][cur_epoch][op] += cc_loss[0].item()
    CC_LOSSES['first_full_cycle'][cur_epoch][op] += cc_loss[1].item()
    CC_LOSSES['second_cycle'][cur_epoch][op] += cc_loss[2].item()
    CC_LOSSES['second_full_cycle'][cur_epoch][op] += cc_loss[3].item()

    ENC_GEN_LOSSES['first_loss'][cur_epoch][op] += enc_gen_loss[0].item()
    ENC_GEN_LOSSES['second_loss'][cur_epoch][op] += enc_gen_loss[1].item()
    ENC_GEN_LOSSES['latent_loss'][cur_epoch][op] += enc_gen_loss[2].item()


def normalize_epoch_loss(scale, cur_epoch, op='train'):
    DISC_LOSSES['first_loss'][cur_epoch][op] /= scale
    DISC_LOSSES['second_loss'][cur_epoch][op] /= scale
    DISC_LOSSES['latent_loss'][cur_epoch][op] /= scale

    CC_LOSSES['first_cycle'][cur_epoch][op] /= scale
    CC_LOSSES['first_full_cycle'][cur_epoch][op] /= scale
    CC_LOSSES['second_cycle'][cur_epoch][op] /= scale
    CC_LOSSES['second_full_cycle'][cur_epoch][op] /= scale

    ENC_GEN_LOSSES['first_loss'][cur_epoch][op] /= scale
    ENC_GEN_LOSSES['second_loss'][cur_epoch][op] /= scale
    ENC_GEN_LOSSES['latent_loss'][cur_epoch][op] /= scale