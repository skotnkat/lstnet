import torch.nn as nn
import torch
import utils

W_1, W_2, W_3, W_4, W_5, W_6, W_l = utils.initialize_weights()
adversarial_loss = nn.BCELoss()
cycle_loss = nn.L1Loss()


def adversarial_loss_real(batch):
    batch_size = batch.size(0)
    ones_labels = torch.ones(batch_size, 1, device=batch.device)  # expecting to be real
    
    return adversarial_loss(batch, ones_labels)
    

def adversarial_loss_gen(batch):
    batch_size = batch.size(0)
    zeros_labels = torch.zeros(batch_size, 1, device=batch.device)  # expecting to be real

    return adversarial_loss(batch, zeros_labels)


def network_adversarial_loss(batch_real, batch_gen):
    real_loss = adversarial_loss_real(batch_real)
    gen_loss = adversarial_loss_gen(batch_gen)

    return real_loss + gen_loss  # normalization


def compute_discriminator_loss(model, first_domain_batch, second_domain_batch):
    # first domain discriminator
    first_domain_real = model.first_discriminator.forward(first_domain_batch)
    first_domain_gen = model.run_second_adversarial_network(second_domain_batch)
    first_disc_loss = network_adversarial_loss(first_domain_real, first_domain_gen)

    # second domain discriminator
    second_domain_real = model.second_discriminator.forward(second_domain_batch)
    second_domain_gen = model.run_first_adversarial_network(first_domain_batch)
    second_disc_loss = network_adversarial_loss(second_domain_real, second_domain_gen)

    # latent discriminator
    first_latent, second_latent = model.run_latent_adversarial_network(first_domain_batch, second_domain_batch)
    latent_disc_loss = network_adversarial_loss(first_latent, second_latent)

    return W_1*first_disc_loss + W_2*second_disc_loss + W_l*latent_disc_loss


def cycle_consistency_loss(model, first_domain_batch, second_domain_batch):
    first_to_first = model.map_first_to_first(first_domain_batch)
    cc_loss_1 = cycle_loss(first_domain_batch, first_to_first)

    first_to_second_to_first = model.map_second_to_first(model.map_first_to_second(first_domain_batch))
    cc_loss_2 = cycle_loss(first_domain_batch, first_to_second_to_first)

    second_to_second = model.map_second_to_second(second_domain_batch)
    cc_loss_3 = cycle_loss(second_domain_batch, second_to_second)

    second_to_first_to_second = model.map_first_to_second(model.map_second_to_first(second_domain_batch))
    cc_loss_4 = cycle_loss(second_domain_batch, second_to_first_to_second)

    return W_3*cc_loss_1 + W_4*cc_loss_2 + W_5*cc_loss_3 + W_6*cc_loss_4


def compute_enc_gen_loss(model, first_domain_batch, second_domain_batch):
    first_gen = model.run_second_adversarial_network(second_domain_batch)
    first_gen_loss = adversarial_loss_real(first_gen)

    second_gen = model.run_first_adversarial_network(first_domain_batch)
    second_gen_loss = adversarial_loss_real(second_gen)

    cc_loss = cycle_consistency_loss(model, first_domain_batch, second_domain_batch)

    return W_1 * first_gen_loss + W_2 * second_gen_loss + cc_loss
