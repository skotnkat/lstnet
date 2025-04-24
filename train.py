import numpy as np
import json
import copy

from models.lstnet import LSTNET
from data_preparation import get_training_loader
import utils
import time

CUR_EPOCH = 0
MAX_PATIENCE = 0
DELTA_LOSS_FRAC = 1e-3


def run_loop(model, loader, op='train'):
    epoch_loss = 0
    for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
        first_real = first_real.to(utils.DEVICE).detach()
        second_real = second_real.to(utils.DEVICE).detach()

        if op == 'val':
            disc_loss, enc_gen_loss, cc_loss = model.run_eval_loop(first_real, second_real)

        # update discriminators
        elif batch_idx % 2 == 0:  # op == train
            disc_loss, enc_gen_loss, cc_loss = model.update_disc(first_real, second_real)

        #############################################################
        # update encoders and generators
        else:
            disc_loss, enc_gen_loss, cc_loss = model.update_enc_gen(first_real, second_real)
        #############################################################

        epoch_loss += sum(disc_loss) + sum(cc_loss)

    epoch_loss /= len(loader)  # compute mean of the losses
    utils.normalize_epoch_loss(len(loader), CUR_EPOCH, op)

    utils.log_epoch_loss(disc_loss, enc_gen_loss, cc_loss, CUR_EPOCH, op)

    return epoch_loss


def train(model, train_loader, val_loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    global CUR_EPOCH

    best_model = None
    best_loss = np.inf
    best_epoch_idx = None
    prev_loss = None

    train_loss_list = []
    val_loss_list = []
    cur_patience = 0
    while True:
        utils.init_epoch_loss()

        start_time = time.time()
        train_loss = run_loop(model, train_loader)
        end_time = time.time()

        train_loss_list.append(train_loss)
        train_time = end_time-start_time

        start_time = time.time()
        val_loss = run_loop(model, val_loader, op='val')
        end_time = time.time()

        val_loss_list.append(val_loss)
        val_time = end_time - start_time

        if prev_loss is not None:
            rel_change = np.abs(val_loss - prev_loss) / prev_loss
            if rel_change < DELTA_LOSS_FRAC:
                print(f'Relative loss change is {rel_change:.5f} < {DELTA_LOSS_FRAC} -> stopping')
                break

        prev_loss = val_loss
        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_loss
            best_epoch_idx = CUR_EPOCH
            cur_patience = 0

        else:
            cur_patience += 1

        print(f'End of epoch {CUR_EPOCH}')
        print(f'\tCurrent train loss: {train_loss}, val loss: {val_loss}')
        print(f'\tTrain took: {(train_time) / 60:.2f} min, Val took: {(val_time) / 60:.2f} min')
        print(f'\tPatience: {cur_patience}')

        # if cur_patience > MAX_PATIENCE:
        #     print(f'\tMax Patience Reached')
        #     break

        CUR_EPOCH += 1

        if CUR_EPOCH % 10 == 0:
            loss_logs = {'disc_loss': utils.DISC_LOSSES, 'enc_gen_loss': utils.ENC_GEN_LOSSES, 'cc_loss': utils.CC_LOSSES,
                         'train_loss': train_loss_list, 'val_loss': val_loss_list}

            with open(f'{utils.OUTPUT_FOLDER}/{utils.LOSS_FILE}.json', 'w') as file:
                json.dump(loss_logs, file)

    print(f'Saving model in epoch {best_epoch_idx} with best val loss: {best_loss}')
    best_model.save_model('best_model.pth')

    loss_logs = {'disc_loss': utils.DISC_LOSSES, 'enc_gen_loss': utils.ENC_GEN_LOSSES, 'cc_loss': utils.CC_LOSSES,
                 'train_loss': train_loss_list, 'val_loss': val_loss_list}
    with open(f'{utils.OUTPUT_FOLDER}{utils.LOSS_FILE}.json', 'w') as file:
        json.dump(loss_logs, file)

    best_model.save_model('last_model.pth')

    return model


def run(first_domain_name, second_domain_name, supervised):
    train_loader, val_loader = get_training_loader(first_domain_name, second_domain_name, supervised)
    print('Creating an instance of LSTNET model')
    model = LSTNET(first_domain_name, second_domain_name)
    print(f'LSTNET model initialized')
    model.to(utils.DEVICE)
    print('LSTNET model moved to device')

    model = train(model, train_loader, val_loader)

    print('Model trained.')
    print('Loss scores dumped.')

    model.to('cpu')

    return model
