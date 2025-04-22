import numpy as np
import torch
from torch.optim import Adam
import json

from models.lstnet import LSTNET
from loss_functions import compute_discriminator_loss, compute_enc_gen_loss, compute_cc_loss
from data_preparation import get_training_loader
import utils
import time
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
    disc_loss_2.backward()
    disc_loss_latent.backward()
    optim_disc_1.step()
    optim_disc_2.step()
    optim_disc_latent.step()

    total_disc_loss = disc_loss_1.item() + disc_loss_2.item() + disc_loss_latent.item()

    with torch.no_grad():
        first_cycle, second_cycle, first_full_cycle, second_full_cycle = get_cc_components(model,
                                                                                           first_gen, second_gen,
                                                                                           first_latent, second_latent)
        cc_loss = compute_cc_loss(first_real, second_real,
                                  first_cycle, second_cycle,
                                  first_full_cycle, second_full_cycle)

    return total_disc_loss + cc_loss.item()


def update_enc_gen(model, first_real, second_real, optim):
    optim.zero_grad()

    second_gen, first_latent = model.map_first_to_second(first_real, return_latent=True)
    first_gen, second_latent = model.map_second_to_first(second_real, return_latent=True)

    enc_gen_loss = compute_enc_gen_loss(model, first_gen, second_gen, first_latent, second_latent)

    # cycle consistency
    first_cycle, second_cycle, first_full_cycle, second_full_cycle = get_cc_components(model,
                                                                                       first_gen, second_gen,
                                                                                       first_latent, second_latent)
    cc_loss = compute_cc_loss(first_real, second_real,
                              first_cycle, second_cycle,
                              first_full_cycle, second_full_cycle)

    enc_gen_loss_total = enc_gen_loss + cc_loss
    enc_gen_loss_total.backward()
    optim.step()

    with torch.no_grad():
        disc_loss_1, disc_loss_2, disc_loss_latent = compute_discriminator_loss(model, first_real, second_real,
                                                                                first_gen.detach(), second_gen.detach(),
                                                                                first_latent.detach(), second_latent.detach())

    total_disc_loss = disc_loss_1.item() + disc_loss_2.item() + disc_loss_latent.item()
    return cc_loss.item() + total_disc_loss


def train(model, loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    optim_disc_1 = Adam(model.first_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_disc_2 = Adam(model.second_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_disc_latent = Adam(model.latent_discriminator.parameters(), lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)
    optim_enc_gen = Adam(model.enc_gen_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)

    converged = False
    prev_avg_loss = np.inf
    loss_list = []
    start_time = time.time()
    while not converged:
        total_loss = 0
        batch_idx = 0
        for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
            first_real = first_real.to(utils.DEVICE).detach()
            second_real = second_real.to(utils.DEVICE).detach()

            #############################################################
            # update discriminators
            if batch_idx % 2 == 0:
                total_loss += update_disc(model, first_real, second_real, optim_disc_1, optim_disc_2, optim_disc_latent)
            #############################################################
            # update encoders and generators
            else:
                total_loss += update_enc_gen(model, first_real, second_real, optim_enc_gen)

            #############################################################

            if batch_idx % 200 == 0:
                end_time = time.time()
                print(f'Batch {batch_idx} processed, took: {(end_time - start_time)/60:.2f} minutes')
                start_time = time.time()
            #############################################################

        cur_avg_loss = total_loss / (batch_idx+1)  # batch_idx starts from 0

        if np.abs(cur_avg_loss - prev_avg_loss) < utils.DELTA_LOSS:
            converged = True

        loss_list.append(cur_avg_loss)
        prev_avg_loss = cur_avg_loss

        print(f'End of epoch {CUR_EPOCH}, current total loss: {cur_avg_loss}')
        torch.save(model.state_dict(), f"output/model_{CUR_EPOCH}.pth")
        CUR_EPOCH += 1

    return model, loss_list


def run(first_domain_name, second_domain_name, supervised):
    data_loader = get_training_loader(first_domain_name, second_domain_name, supervised)
    model = LSTNET(first_domain_name, second_domain_name)
    model.to(utils.DEVICE)
    print('LSTNET model initialized')

    model, loss_list = train(model, data_loader)

    with open(f'{utils.OUTPUT_FOLDER}/{utils.LOSS_FILE}.json', 'a') as file:
        json.dump(loss_list, file, indent=2)

    print('Model trained.')
    print('Loss scores dumped.')

    model.to('cpu')

    return model
