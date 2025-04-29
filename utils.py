import json

PARAMS_FILE_PATH = None
NETWORK_NAMES = {"first_encoder", "second_encoder", "shared_encoder", "first_generator", "second_generator",
                 "shared_generator", "first_discriminator", "second_discriminator", "latent_discriminator"}

OUTPUT_FOLDER = None
BATCH_SIZE = None
ADAM_LR = None
ADAM_DECAY = None
DELTA_LOSS = None
MANUAL_SEED = None
VAL_SIZE = None

DEVICE = None
NUM_WORKERS = None

FIRST_INPUT_SHAPE, SECOND_INPUT_SHAPE = None, None
FIRST_IN_CHANNELS_NUM, SECOND_IN_CHANNELS_NUM = None, None

DISC_LOSSES = {'first_loss': [], 'second_loss': [], 'latent_loss': []}
CC_LOSSES = {'first_cycle': [], 'second_cycle': [], 'first_full_cycle': [], 'second_full_cycle': []}
# ENC_GEN_LOSSES = {'first_loss': [], 'second_loss': [], 'latent_loss': []}


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
    DISC_LOSSES['first_loss'].append(0)
    DISC_LOSSES['second_loss'].append(0)
    DISC_LOSSES['latent_loss'].append(0)

    CC_LOSSES['first_cycle'].append(0)
    CC_LOSSES['second_cycle'].append(0)
    CC_LOSSES['first_full_cycle'].append(0)
    CC_LOSSES['second_full_cycle'].append(0)

    # ENC_GEN_LOSSES['first_loss'].append(0)
    # ENC_GEN_LOSSES['second_loss'].append(0)
    # ENC_GEN_LOSSES['latent_loss'].append(0)


def log_epoch_loss(disc_loss, cc_loss, cur_epoch):
    DISC_LOSSES['first_loss'][cur_epoch] += disc_loss[0]
    DISC_LOSSES['second_loss'][cur_epoch] += disc_loss[1]
    DISC_LOSSES['latent_loss'][cur_epoch] += disc_loss[2]

    CC_LOSSES['first_cycle'][cur_epoch] += cc_loss[0]
    CC_LOSSES['second_cycle'][cur_epoch] += cc_loss[1]
    CC_LOSSES['first_full_cycle'][cur_epoch] += cc_loss[2]
    CC_LOSSES['second_full_cycle'][cur_epoch] += cc_loss[3]

    # ENC_GEN_LOSSES['first_loss'][cur_epoch] += enc_gen_loss[0]
    # ENC_GEN_LOSSES['second_loss'][cur_epoch] += enc_gen_loss[1]
    # ENC_GEN_LOSSES['latent_loss'][cur_epoch] += enc_gen_loss[2]


def normalize_epoch_loss(scale, cur_epoch):
    DISC_LOSSES['first_loss'][cur_epoch] /= scale
    DISC_LOSSES['second_loss'][cur_epoch] /= scale
    DISC_LOSSES['latent_loss'][cur_epoch] /= scale

    CC_LOSSES['first_cycle'][cur_epoch] /= scale
    CC_LOSSES['first_full_cycle'][cur_epoch] /= scale
    CC_LOSSES['second_cycle'][cur_epoch] /= scale
    CC_LOSSES['second_full_cycle'][cur_epoch] /= scale

    # ENC_GEN_LOSSES['first_loss'][cur_epoch] /= scale
    # ENC_GEN_LOSSES['second_loss'][cur_epoch] /= scale
    # ENC_GEN_LOSSES['latent_loss'][cur_epoch] /= scale


def check_file_ending(file, ending):
    if not file.endswith(ending):
        file = file + ending

    return file


def set_input_dimensions(dataset):
    global FIRST_INPUT_SHAPE, SECOND_INPUT_SHAPE, FIRST_IN_CHANNELS_NUM, SECOND_IN_CHANNELS_NUM
    first_img, _, second_img, _ = dataset.__getitem__(0)

    FIRST_INPUT_SHAPE = first_img.shape[1:]
    FIRST_IN_CHANNELS_NUM = first_img.shape[0]

    SECOND_INPUT_SHAPE = second_img.shape[1:]
    SECOND_IN_CHANNELS_NUM = second_img.shape[0]
