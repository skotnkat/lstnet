import numpy as np
import json
import copy

from models.lstnet import LSTNET
from data_preparation import get_training_loader
import utils
import time
from tqdm import tqdm

MAX_PATIENCE = None

MODEL_PATH = "lstnet.pth"
LOSS_FILE = "loss_logs.json"


def run_loop(model, loader, train_op=True):
    epoch_loss = 0

    for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
        first_real = first_real.to(utils.DEVICE).detach()
        second_real = second_real.to(utils.DEVICE).detach()
        if train_op:
            disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = model.run_eval_loop(first_real, second_real)
        elif batch_idx % 2 == 0:
            disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = model.update_disc(first_real, second_real)
        else:
            disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = model.update_enc_gen(first_real, second_real)

        epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)
        utils.log_epoch_loss(disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple, train_op)

    scale = len(loader)
    utils.normalize_epoch_loss(scale, train_op)
    epoch_loss /= scale

    return epoch_loss


def train_and_validate(model, train_loader, max_epoch_num, val_loader=None):
    """
        First phase of training. Without knowledge of the labels (will be ignoring the labels).
        Validate only if val_loader is passed.
    """

    best_model = None
    best_loss = np.inf
    best_epoch_idx = None
    train_loss_list = []
    val_loss_list = []
    cur_patience = 0

    model.to(utils.DEVICE)
    for epoch_idx in tqdm(range(max_epoch_num)):
        start_time = time.time()
        utils.init_epoch_loss()
        epoch_loss = run_loop(model, train_loader)
        train_loss_list.append(epoch_loss)

        if val_loader is not None:  # if validation is being run then the decision loss is validation, otherwise train  `
            epoch_loss = run_loop(model, val_loader, train_op=False)
            val_loss_list.append(epoch_loss)


        if epoch_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = epoch_loss
            best_epoch_idx = epoch_idx
            cur_patience = 0

        else:
            cur_patience += 1

            if cur_patience >= MAX_PATIENCE:
                print(f'max patience reached')
                break

        end_time = time.time()
        print(f'End of epoch {epoch_idx}')
        print(f'\tCurrent val loss: {epoch_loss}')
        print(f'\tEpoch took: {(end_time - start_time) / 60:.2f} min')
        print(f'\tPatience: {cur_patience}')

    print(f'Best loss obtained in epoch {best_epoch_idx}')

    utils.LOSS_LOGS['best_epoch_idx'] = best_epoch_idx

    with open(f'{utils.OUTPUT_FOLDER}{LOSS_FILE}', 'w') as file:
        json.dump(utils.LOSS_LOGS, file, indent=2)

    best_model.to("cpu")
    return best_model, best_epoch_idx


def run_full_training(first_domain_name, second_domain_name, supervised, epoch_num):
    loader = get_training_loader(first_domain_name, second_domain_name, supervised, split_data=False)
    model = LSTNET(first_domain_name, second_domain_name)
    utils.init_logs(['train'])

    print('Starting full training')
    model, best_epoch_idx = train_and_validate(model, loader, epoch_num)
    print('Model trained on full train dataset.')
    model_path = f'{utils.OUTPUT_FOLDER}{MODEL_PATH}'
    model.save_model(model_path)

    return model


def run(first_domain_name, second_domain_name, supervised, epoch_num):
    train_loader, val_loader = get_training_loader(first_domain_name, second_domain_name, supervised, split_data=True)
    model = LSTNET(first_domain_name, second_domain_name)
    utils.init_logs(['train', 'val'])

    print('Starting train and validate')
    model, best_epoch_idx = train_and_validate(model, train_loader, val_loader, epoch_num)  # pass both loaders at once
    print('Model trained trained and validated.')

    model_path = f'{utils.OUTPUT_FOLDER}{MODEL_PATH}'
    model.save_model(model_path)

    return model
