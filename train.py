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

    optim_enc_dec = Adam(enc_gen_params, lr=utils.ADAM_LR, betas=utils.ADAM_DECAY)

    epoch_num = 0
    converged = False
    prev_total_loss = np.inf
    loss_list = []
    while not converged:
        total_disc_loss, total_enc_gen_loss = None, None
        for batch_idx, (first_domain_batch, _, second_domain_batch, _) in enumerate(loader):
            total_disc_loss, total_enc_gen_loss = compute_loss(model, first_domain_batch, second_domain_batch)

            if batch_idx % 2:  # odd -> update discriminators
                optim_disc.zero_grad()
                total_disc_loss.backward()
                optim_disc.step()

            else:  # update encoders and generators
                optim_enc_dec.zero_grad()
                total_enc_gen_loss.backward()
                optim_enc_dec.step()

            if (batch_idx % 100 == 0):
                print(f'batch idx: {batch_idx}, disc loss: {total_disc_loss}, enc gen loss: {total_enc_gen_loss}')

        cur_total_loss = total_disc_loss + total_enc_gen_loss

        if torch.abs(cur_total_loss - prev_total_loss) < utils.DELTA_LOSS:
            converged = True

        loss_list.append(cur_total_loss)
        prev_total_loss = cur_total_loss

        print(f'End of epoch {epoch_num}, current total loss: {cur_total_loss}')
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
