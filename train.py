import numpy as np
import json
import copy

from models.lstnet import LSTNET
from data_preparation import get_training_loader
import utils
import time

CUR_EPOCH = 0
MAX_PATIENCE = None
DELTA_LOSS = 1e-3

MODEL_PATH = "lstnet.pth"
LOSS_FILE = "loss_logs.json"


def check_stop_condition(cur_loss, prev_loss, cur_patience):
    if prev_loss is not None:
        rel_change = np.abs(cur_loss - prev_loss) / prev_loss

        if rel_change < DELTA_LOSS:
            cur_patience += 1
            print(f'Relative loss change is {rel_change:.5f} < {DELTA_LOSS} -> increasing patience to {cur_patience}')

        else:
            cur_patience = 0
            print(f'Relative loss change not enough: {rel_change:.5f}')

    stop_flag = False
    if cur_patience >= MAX_PATIENCE:
        stop_flag = True

    return stop_flag, cur_patience


def run_train_loop(model, train_loader, run_val=False, val_loader=None):
    for batch_idx, (first_real, _, second_real, _) in enumerate(train_loader):
        first_real = first_real.to(utils.DEVICE).detach()
        second_real = second_real.to(utils.DEVICE).detach()

        if batch_idx % 2 == 0:
            model.update_disc(first_real, second_real)

        else:
            model.update_enc_gen(first_real, second_real)

    if not run_val:
        return

    # validation loop
    if val_loader is None:
        raise ValueError("Validation cannot be run without a loader with validation data.")


    epoch_loss = 0
    for batch_idx, (first_real, _, second_real, _) in enumerate(val_loader):
        first_real = first_real.to(utils.DEVICE)
        second_real = second_real.to(utils.DEVICE)
        disc_loss_tuple, cc_loss_tuple = model.run_eval_loop(first_real, second_real)
        utils.log_epoch_loss(disc_loss_tuple, cc_loss_tuple, CUR_EPOCH)

        epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

    scale = len(val_loader)
    utils.normalize_epoch_loss(scale, CUR_EPOCH)
    epoch_loss /= scale

    return epoch_loss


def run_full_training(first_domain_name, second_domain_name, supervised, epoch_num):
    loader = get_training_loader(first_domain_name, second_domain_name, supervised, data_split=False)
    model = LSTNET(first_domain_name, second_domain_name)
    model.to(utils.DEVICE)

    for idx in range(epoch_num):
        run_train_loop(model, loader)  # without validation, only train

    model_path = f'{utils.OUTPUT_FOLDER}{MODEL_PATH}'
    model.save_model(model_path)

    model.to("cpu")
    return model


def train_and_validate(model, train_loader, val_loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    global CUR_EPOCH

    best_model = None
    best_loss = np.inf
    best_epoch_idx = None

    prev_loss = None
    loss_list = []
    cur_patience = 0

    while True:
        start_time = time.time()
        utils.init_epoch_loss()
        epoch_loss = run_train_loop(model, train_loader, run_val=True, val_loader=val_loader)

        loss_list.append(epoch_loss)

        stop_flag, cur_patience = check_stop_condition(epoch_loss, prev_loss, cur_patience)
        if stop_flag:
            break

        prev_loss = epoch_loss

        if epoch_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = epoch_loss
            best_epoch_idx = CUR_EPOCH

        end_time = time.time()
        print(f'End of epoch {CUR_EPOCH}')
        print(f'\tCurrent val loss: {epoch_loss}')
        print(f'\tEpoch took: {(end_time - start_time) / 60:.2f} min')
        print(f'\tPatience: {cur_patience}')

        CUR_EPOCH += 1

        if CUR_EPOCH % 10 == 0:
            loss_logs = {'disc_loss': utils.DISC_LOSSES, 'cc_loss': utils.CC_LOSSES, 'train_loss': loss_list}  # 'enc_gen_loss': utils.ENC_GEN_LOSSES

            with open(f'{utils.OUTPUT_FOLDER}{LOSS_FILE}', 'w') as file:
                json.dump(loss_logs, file, indent=2)

    print(f'Saving model in epoch {best_epoch_idx} with best val loss: {best_loss}')

    loss_logs = {'disc_loss': utils.DISC_LOSSES, 'cc_loss': utils.CC_LOSSES,
                 'train_loss': loss_list, 'best_epoch_idx': best_epoch_idx}  # 'enc_gen_loss': utils.ENC_GEN_LOSSES,

    with open(f'{utils.OUTPUT_FOLDER}{LOSS_FILE}', 'w') as file:
        json.dump(loss_logs, file, indent=2)

    return best_model, best_epoch_idx


def run(first_domain_name, second_domain_name, supervised):
    train_loader, val_loader = get_training_loader(first_domain_name, second_domain_name, supervised, split_data=True)

    model = LSTNET(first_domain_name, second_domain_name)
    model.to(utils.DEVICE)

    model, best_epoch_idx = train_and_validate(model, train_loader, val_loader)  # pass both loaders at once
    model.to("cpu")

    print('Model trained trained and validated.')
    print('Validation loss scores dumped.')

    model_path = f'{utils.OUTPUT_FOLDER}{MODEL_PATH}'
    model.save_model(model_path)

    return model, best_epoch_idx
