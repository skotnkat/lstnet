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



def train(model, loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    global CUR_EPOCH

    best_model = None
    best_loss = np.inf
    best_epoch_idx = None

    prev_loss = None
    loss_list = []
    cur_patience = 0

    while True:
        utils.init_epoch_loss()
        epoch_loss = 0

        start_time = time.time()
        for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
            first_real = first_real.to(utils.DEVICE).detach()
            second_real = second_real.to(utils.DEVICE).detach()

            #############################################################
            # update discriminators
            if batch_idx % 2 == 0:
                disc_loss, enc_gen_loss, cc_loss = model.update_disc(first_real, second_real)

            # update encoders and generators
            else:
                disc_loss, enc_gen_loss, cc_loss = model.update_enc_gen(first_real, second_real)
            #############################################################

            epoch_loss += sum(disc_loss) + sum(cc_loss)

        epoch_loss /= len(loader)  # compute mean of the losses

        utils.normalize_epoch_loss(len(loader), CUR_EPOCH)
        utils.log_epoch_loss(disc_loss, enc_gen_loss, cc_loss, CUR_EPOCH)
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
        print(f'\tCurrent train loss: {epoch_loss}')
        print(f'\tTrain took: {(end_time - start_time) / 60:.2f} min')
        print(f'\tPatience: {cur_patience}')

        CUR_EPOCH += 1

        if CUR_EPOCH % 10 == 0:
            loss_logs = {'disc_loss': utils.DISC_LOSSES, 'cc_loss': utils.CC_LOSSES, 'train_loss': loss_list}  # 'enc_gen_loss': utils.ENC_GEN_LOSSES

            with open(f'{utils.OUTPUT_FOLDER}{utils.LOSS_FILE}', 'w') as file:
                json.dump(loss_logs, file, indent=2)

    print(f'Saving model in epoch {best_epoch_idx} with best val loss: {best_loss}')

    loss_logs = {'disc_loss': utils.DISC_LOSSES, 'cc_loss': utils.CC_LOSSES,
                 'train_loss': loss_list, 'best_epoch_idx': best_epoch_idx}  # 'enc_gen_loss': utils.ENC_GEN_LOSSES,

    with open(f'{utils.OUTPUT_FOLDER}{utils.LOSS_FILE}', 'w') as file:
        json.dump(loss_logs, file, indent=2)

    return best_model


def run(first_domain_name, second_domain_name, supervised, output_model_file):
    loader = get_training_loader(first_domain_name, second_domain_name, supervised)
    print('Creating an instance of LSTNET model')
    model = LSTNET(first_domain_name, second_domain_name)
    print(f'LSTNET model initialized')
    model.to(utils.DEVICE)
    print('LSTNET model moved to device')

    model = train(model, loader)

    print('Model trained.')
    print('Loss scores dumped.')
    model.to("cpu")

    model_path = f'{utils.OUTPUT_FOLDER}{output_model_file}'
    model.save_model(model_path)

    return model