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

LOSS_LOGS = dict()


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


def init_logs(ops=['train', 'val']):
    for op in ops:
        LOSS_LOGS[op] = {
            'disc_loss': {'first_loss': [], 'second_loss': [], 'latent_loss': []},
            'enc_gen_loss': {'first_loss': [], 'second_loss': [], 'latent_loss': []},
            'cc_loss': {'first_cycle_loss': [], 'second_cycle_loss': [], 'first_full_cycle_loss': [], 'second_full_cycle_loss': []}
        }


def init_epoch_loss(op='train'):
    op_logs = LOSS_LOGS[op]

    op_logs['disc_loss']['first_loss'].append(0)
    op_logs['disc_loss']['second_loss'].append(0)
    op_logs['disc_loss']['latent_loss'].append(0)

    op_logs['enc_gen_loss']['first_loss'].append(0)
    op_logs['enc_gen_loss']['second_loss'].append(0)
    op_logs['enc_gen_loss']['latent_loss'].append(0)

    op_logs['cc_loss']['first_cycle_loss'].append(0)
    op_logs['cc_loss']['second_cycle_loss'].append(0)
    op_logs['cc_loss']['first_full_cycle_loss'].append(0)
    op_logs['cc_loss']['second_full_cycle_loss'].append(0)


def log_epoch_loss(disc_loss, enc_gen_loss, cc_loss, op):
    op_logs = LOSS_LOGS[op]
    cur_epoch = len(op_logs['disc_loss']['first_loss']) - 1  # last epoch

    op_logs['disc_loss']['first_loss'][cur_epoch] += disc_loss[0]
    op_logs['disc_loss']['second_loss'][cur_epoch] += disc_loss[1]
    op_logs['disc_loss']['latent_loss'][cur_epoch] += disc_loss[2]

    op_logs['enc_gen_loss']['first_loss'][cur_epoch] += enc_gen_loss[0]
    op_logs['enc_gen_loss']['second_loss'][cur_epoch] += enc_gen_loss[1]
    op_logs['enc_gen_loss']['latent_loss'][cur_epoch] += enc_gen_loss[2]

    op_logs['cc_loss']['first_cycle_loss'][cur_epoch] += cc_loss[0]
    op_logs['cc_loss']['second_cycle_loss'][cur_epoch] += cc_loss[1]
    op_logs['cc_loss']['first_full_cycle_loss'][cur_epoch] += cc_loss[2]
    op_logs['cc_loss']['second_full_cycle_loss'][cur_epoch] += cc_loss[3]


def normalize_epoch_loss(scale, op):
    op_logs = LOSS_LOGS[op]
    cur_epoch = len(op_logs['disc_loss']['first_loss']) - 1  # last epoch

    op_logs['disc_loss']['first_loss'][cur_epoch] /= scale
    op_logs['disc_loss']['second_loss'][cur_epoch] /= scale
    op_logs['disc_loss']['latent_loss'][cur_epoch] /= scale

    op_logs['enc_gen_loss']['first_loss'][cur_epoch] /= scale
    op_logs['enc_gen_loss']['second_loss'][cur_epoch] /= scale
    op_logs['enc_gen_loss']['latent_loss'][cur_epoch] /= scale

    op_logs['cc_loss']['first_cycle_loss'][cur_epoch] /= scale
    op_logs['cc_loss']['first_full_cycle_loss'][cur_epoch] /= scale
    op_logs['cc_loss']['second_cycle_loss'][cur_epoch] /= scale
    op_logs['cc_loss']['second_full_cycle_loss'][cur_epoch] /= scale




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
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assign_device():
    global DEVICE

    DEVICE = get_device()
    print(f'Using device: {DEVICE}')
