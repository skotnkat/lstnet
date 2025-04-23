import torch.nn as nn
import torch
import utils

W_1, W_2, W_3, W_4, W_5, W_6, W_l = utils.initialize_weights()
adversarial_loss = nn.BCEWithLogitsLoss()
cycle_loss = nn.L1Loss()


def adversarial_loss_real(batch):
    batch_size = batch.size(0)
    ones_labels = torch.ones(batch_size, 1, device=batch.device, requires_grad=False)  # expecting to be real
    
    return adversarial_loss(batch, ones_labels)
    

def adversarial_loss_gen(batch):
    batch_size = batch.size(0)
    zeros_labels = torch.zeros(batch_size, 1, device=batch.device, requires_grad=False)  # expecting to be fake

    return adversarial_loss(batch, zeros_labels)


def network_adversarial_loss(batch_real, batch_gen):
    real_loss = adversarial_loss_real(batch_real)
    gen_loss = adversarial_loss_gen(batch_gen)

    return real_loss + gen_loss  # normalization


def compute_discriminator_loss(model, first_real, second_real, first_gen, second_gen, first_latent, second_latent):
    first_real_disc = model.first_discriminator.forward(first_real)
    first_gen_disc = model.first_discriminator.forward(first_gen)
    first_disc_loss = network_adversarial_loss(first_real_disc, first_gen_disc)

    second_real_disc = model.second_discriminator.forward(second_real)
    second_gen_disc = model.second_discriminator.forward(second_gen)
    second_disc_loss = network_adversarial_loss(second_real_disc, second_gen_disc)

    first_latent_disc = model.latent_discriminator.forward(first_latent)
    second_latent_disc = model.latent_discriminator.forward(second_latent)
    latent_disc_loss = network_adversarial_loss(first_latent_disc, second_latent_disc)

    return W_1*first_disc_loss, W_2*second_disc_loss, W_l*latent_disc_loss


def compute_cc_loss(first_real, second_real, first_cycle, second_cycle, first_full_cycle, second_full_cycle):
    cc_loss_1 = cycle_loss(first_cycle, first_real)  # (predictions, target)
    cc_loss_2 = cycle_loss(second_cycle, second_real)

    cc_loss_3 = cycle_loss(first_full_cycle, first_real)
    cc_loss_4 = cycle_loss(second_full_cycle, second_real)

    return W_3*cc_loss_1 + W_4*cc_loss_2 + W_5*cc_loss_3 + W_6*cc_loss_4


def compute_enc_gen_loss(model, first_gen, second_gen, first_latent, second_latent):
    first_gen_disc = model.first_discriminator.forward(first_gen)
    first_gen_loss = adversarial_loss_real(first_gen_disc)

    second_gen_disc = model.second_discriminator.forward(second_gen)
    second_gen_loss = adversarial_loss_real(second_gen_disc)

    first_latent_disc = model.latent_discriminator.forward(first_latent)
    second_latent_disc = model.latent_discriminator.forward(second_latent)
    latent_loss = network_adversarial_loss(second_latent_disc, first_latent_disc) / 2

    return W_1*first_gen_loss, W_2*second_gen_loss, W_l*latent_loss

