import numpy as np
import json
import copy

from models.lstnet import LSTNET
from data_preparation import get_training_loader
import utils
import time

CUR_EPOCH = 0









def train(model, loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    global CUR_EPOCH
    optim_disc_1 = Adam(model.first_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_disc_2 = Adam(model.second_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_disc_latent = Adam(model.latent_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_enc_gen = Adam(model.enc_gen_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)

    converged = False
    prev_epoch_loss = np.inf
    best_epoch_loss = np.inf
    best_weights = None
    best_epoch_idx = np.inf

    loss_list = []
    start_time = time.time()


    while not converged:
        DISC_LOSSES[CUR_EPOCH] = []
        CC_LOSSES[CUR_EPOCH] = []
        ENC_GEN_LOSSES[CUR_EPOCH] = []

        epoch_loss = 0
        for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
            first_real = first_real.to(utils.DEVICE).detach()
            second_real = second_real.to(utils.DEVICE).detach()

            #############################################################
            # update discriminators
            if batch_idx % 2 == 0:
                epoch_loss += update_disc(model, first_real, second_real, optim_disc_1, optim_disc_2, optim_disc_latent)
            #############################################################
            # update encoders and generators
            else:
                epoch_loss += update_enc_gen(model, first_real, second_real, optim_enc_gen)

            #############################################################

        epoch_loss /= len(loader)  # compute mean of the losses

        if np.abs(epoch_loss - prev_epoch_loss) < utils.DELTA_LOSS:
            converged = True

        loss_list.append(epoch_loss)

        # should be last but also best?
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch_idx = CUR_EPOCH

        end_time = time.time()
        print(f'End of epoch {CUR_EPOCH}')
        print(f'\tCurrent total loss: {epoch_loss}')
        print(f'\tTook: {(end_time-start_time)/60:.2f} min')

        CUR_EPOCH += 1
        start_time = time.time()

        if CUR_EPOCH % 10 == 0:
            torch.save(best_weights, f"model_weights_{CUR_EPOCH}.pth")
            loss_logs = {'disc_loss': DISC_LOSSES, 'enc_gen_loss': ENC_GEN_LOSSES, 'cc_loss': CC_LOSSES,
                         'epoch_loss': loss_list}

            with open(f'{utils.OUTPUT_FOLDER}/loss_logs.json', 'w') as file:
                json.dump(loss_logs, file)

    print(f'Best epoch: {best_epoch_idx}')
    torch.save(best_weights, "best_model_weights.pth")

    return model, loss_list


def run(first_domain_name, second_domain_name, supervised):
    data_loader = get_training_loader(first_domain_name, second_domain_name, supervised)
    print('Creating an instance of LSTNET model')
    model = LSTNET(first_domain_name, second_domain_name)
    print(f'LSTNET model initialized')
    model.to(utils.DEVICE)
    print('LSTNET model moved to device')

    model, loss_list = train(model, data_loader)

    with open(f'{utils.OUTPUT_FOLDER}/{utils.LOSS_FILE}.json', 'w') as file:
        json.dump(loss_list, file, indent=2)

    print('Model trained.')
    print('Loss scores dumped.')

    model.to('cpu')

    return model
