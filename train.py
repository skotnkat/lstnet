import numpy as np
import torch
from torch.optim import Adam
import json

from models.lstnet import LSTNET
from loss_functions import compute_discriminator_loss, compute_enc_gen_loss, compute_cc_loss
from data_preparation import get_training_loader
import utils


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


def run_training(model, loader):
    """First phase of training. Without knowledge of the labels (will be ignoring the labels)."""
    disc_params = list(model.first_discriminator.parameters()) \
                  + list(model.second_discriminator.parameters()) \
                  + list(model.latent_discriminator.parameters())
    optim_disc = Adam(disc_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)

    enc_gen_params = list(model.first_encoder.parameters()) \
                          + list(model.second_encoder.parameters()) \
                          + list(model.shared_encoder.parameters()) \
                          + list(model.first_generator.parameters()) \
                          + list(model.second_generator.parameters()) \
                          + list(model.shared_generator.parameters())

    optim_enc_gen = Adam(enc_gen_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)

    epoch_num = 0
    converged = False
    prev_avg_loss = np.inf
    loss_list = []
    while not converged:
        total_loss = 0
        batch_idx = 0
        for batch_idx, (first_real, _, second_real, _) in enumerate(loader):
            #############################################################
            # update discriminators
            optim_disc.zero_grad()

            second_gen, first_latent = model.map_first_to_second(first_real, return_latent=True)
            first_gen, second_latent = model.map_second_to_first(second_real, return_latent=True)

            disc_loss = compute_discriminator_loss(model,
                                                   first_real, second_real,
                                                   first_gen, second_gen,
                                                   first_latent, second_latent)

            disc_loss.backward(retain_graph=True)
            optim_disc.step()

            total_loss += disc_loss

            #############################################################
            # update encoders and generators
            optim_enc_gen.zero_grad()

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
            optim_enc_gen.step()

            total_loss += cc_loss
            #############################################################

        cur_avg_loss = total_loss / batch_idx

        if torch.abs(cur_avg_loss - prev_avg_loss) < utils.DELTA_LOSS:
            converged = True

        loss_list.append(cur_avg_loss)
        prev_avg_loss = cur_avg_loss

        print(f'End of epoch {epoch_num}, current total loss: {cur_avg_loss}')
        epoch_num += 1

    return model, loss_list


def train(first_domain_name, second_domain_name, supervised):
    data_loader = get_training_loader(first_domain_name, second_domain_name, supervised)
    model = LSTNET().to(utils.DEVICE)
    print('LSTNet model initialized')

    model, loss_list = run_training(model, data_loader)

    with open(f'{utils.OUTPUT_FOLDER}/{utils.LOSS_FILE}.json', 'w') as file:
        json.dump(loss_list, file, indent=2)

    print('Model trained.')
    print('Loss scores dumped.')
    return model
