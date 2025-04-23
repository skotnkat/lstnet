import numpy as np
import torch
from torch.optim import Adam
import json
import copy

from models.lstnet import LSTNET
from loss_functions import compute_discriminator_loss, compute_enc_gen_loss, compute_cc_loss
from data_preparation import get_training_loader
import utils
import time

from utils import OUTPUT_FOLDER

DISC_LOSSES = dict()
CC_LOSSES = dict()
ENC_GEN_LOSSES = dict()

CUR_EPOCH = 0


def get_cc_components(model, first_gen, second_gen, first_latent, second_latent):
    # map latent representation of real first images back to first domain
    first_cycle = model.map_latent_to_first(first_latent)

    # map latent representation of real second images back to second domain
    second_cycle = model.map_latent_to_second(second_latent)

    # map generated images in second domain back to first domain
    first_full_cycle = model.map_second_to_first(second_gen)

    # map generated images in first domain back to second domain
    second_full_cycle = model.map_first_to_second(first_gen)

    return first_cycle, second_cycle, first_full_cycle, second_full_cycle


def update_disc(model, first_real, second_real, optim_disc_1, optim_disc_2, optim_disc_latent):
    optim_disc_1.zero_grad()
    optim_disc_2.zero_grad()
    optim_disc_latent.zero_grad()

    with torch.no_grad():
        second_gen, first_latent = model.map_first_to_second(first_real, return_latent=True)
        first_gen, second_latent = model.map_second_to_first(second_real, return_latent=True)

        first_gen = first_gen.detach()
        second_gen = second_gen.detach()

        first_latent = first_latent.detach()
        second_latent = second_latent.detach()

    disc_loss_1, disc_loss_2, disc_loss_latent = compute_discriminator_loss(model, first_real, second_real,
                                                                            first_gen, second_gen,
                                                                            first_latent, second_latent)

    disc_loss_1.backward()
    optim_disc_1.step()

    disc_loss_2.backward()
    optim_disc_2.step()

    disc_loss_latent.backward()
    optim_disc_latent.step()


    DISC_LOSSES[CUR_EPOCH].append({'first_loss' : disc_loss_1.item(),
                                   'second_loss' : disc_loss_2.item(),
                                   'latent_loss': disc_loss_latent.item()})

    with torch.no_grad():
        first_cycle, second_cycle, first_full_cycle, second_full_cycle = get_cc_components(model,
                                                                                           first_gen, second_gen,
                                                                                           first_latent, second_latent)
        cc_loss = compute_cc_loss(first_real, second_real,
                                  first_cycle, second_cycle,
                                  first_full_cycle, second_full_cycle)


    CC_LOSSES[CUR_EPOCH].append(cc_loss.item())

    return disc_loss_1.item() + disc_loss_2.item() + disc_loss_latent.item() + cc_loss.item()


def update_enc_gen(model, first_real, second_real, optim):
    optim.zero_grad()

    second_gen, first_latent = model.map_first_to_second(first_real, return_latent=True)
    first_gen, second_latent = model.map_second_to_first(second_real, return_latent=True)

    first_enc_gen_loss, second_enc_gen_loss, latent_enc_gen_loss = compute_enc_gen_loss(model, first_gen, second_gen, first_latent, second_latent)

    # cycle consistency
    first_cycle, second_cycle, first_full_cycle, second_full_cycle = get_cc_components(model,
                                                                                       first_gen, second_gen,
                                                                                       first_latent, second_latent)
    cc_loss = compute_cc_loss(first_real, second_real,
                              first_cycle, second_cycle,
                              first_full_cycle, second_full_cycle)

    enc_gen_loss_total = first_enc_gen_loss + second_enc_gen_loss + latent_enc_gen_loss + cc_loss
    enc_gen_loss_total.backward()

    optim.step()
    CC_LOSSES[CUR_EPOCH].append(cc_loss.item())
    ENC_GEN_LOSSES[CUR_EPOCH].append({'first_loss': first_enc_gen_loss.item(), 'second_loss': second_enc_gen_loss.item(), 'latent_loss' : latent_enc_gen_loss.item()})


    with torch.no_grad():
        disc_loss_1, disc_loss_2, disc_loss_latent = compute_discriminator_loss(model, first_real, second_real,
                                                                                first_gen.detach(), second_gen.detach(),
                                                                                first_latent.detach(), second_latent.detach())

    DISC_LOSSES[CUR_EPOCH].append({'first_loss': disc_loss_1.item(), 'second_loss': disc_loss_2.item(), 'latent_loss': disc_loss_latent.item()})

    return cc_loss.item() + disc_loss_1.item() + disc_loss_2.item() + disc_loss_latent.item()



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
                         'epoch_loss': epoch_loss}

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
