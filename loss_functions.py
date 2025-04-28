import torch.nn as nn
import torch
import utils

W_1, W_2, W_3, W_4, W_5, W_6, W_l = utils.initialize_weights()
adversarial_loss = nn.BCEWithLogitsLoss()
cycle_loss = nn.L1Loss()


def adversarial_loss_real(batch, label_smoothing=False):
    batch_size = batch.size(0)
    ones_labels = torch.ones(batch_size, 1, device=batch.device, requires_grad=False)  # expecting to be real
    
    # if label_smoothing:
    #     ones_labels = torch.full((batch_size, 1), 0.9, device=batch.device, requires_grad=False)  # avoid discriminators to become too strong

    return adversarial_loss(batch, ones_labels)


def adversarial_loss_gen(batch, label_smoothing=False):
    batch_size = batch.size(0)
    zeros_labels = torch.zeros(batch_size, 1, device=batch.device, requires_grad=False)  # expecting to be fake

    # if label_smoothing:
    #     zeros_labels = torch.full((batch_size, 1), 0.9, device=batch.device, requires_grad=False)  # avoid discriminators to become too strong

    return adversarial_loss(batch, zeros_labels)


def network_adversarial_loss(batch_real, batch_gen, label_smoothing=False):
    real_loss = adversarial_loss_real(batch_real, label_smoothing)
    gen_loss = adversarial_loss_gen(batch_gen, label_smoothing)

    return real_loss + gen_loss


def compute_discriminator_loss(model, first_real, second_real, first_gen, second_gen, first_latent, second_latent,
                               return_grad=True):
    first_real_disc = model.first_discriminator.forward(first_real)
    first_gen_disc = model.first_discriminator.forward(first_gen)
    first_disc_loss = network_adversarial_loss(first_real_disc, first_gen_disc, label_smoothing=True)

    second_real_disc = model.second_discriminator.forward(second_real)
    second_gen_disc = model.second_discriminator.forward(second_gen)
    second_disc_loss = network_adversarial_loss(second_real_disc, second_gen_disc, label_smoothing=True)

    first_latent_disc = model.latent_discriminator.forward(first_latent)
    second_latent_disc = model.latent_discriminator.forward(second_latent)
    latent_disc_loss = network_adversarial_loss(first_latent_disc, second_latent_disc, label_smoothing=True)

    if return_grad:
        return W_1 * first_disc_loss, W_2 * second_disc_loss, W_l * latent_disc_loss

    return W_1 * first_disc_loss.item(), W_2 * second_disc_loss.item(), W_l * latent_disc_loss.item()


def compute_cc_loss(first_real, second_real, first_cycle, second_cycle, first_full_cycle, second_full_cycle,
                    return_grad=True):
    cc_loss_1 = cycle_loss(first_cycle, first_real)  # (predictions, target)
    cc_loss_2 = cycle_loss(second_cycle, second_real)

    cc_loss_3 = cycle_loss(first_full_cycle, first_real)
    cc_loss_4 = cycle_loss(second_full_cycle, second_real)

    if return_grad:
        return W_3 * cc_loss_1, W_4 * cc_loss_2, W_5 * cc_loss_3, W_6 * cc_loss_4

    return W_3 * cc_loss_1.item(), W_4 * cc_loss_2.item(), W_5 * cc_loss_3.item(), W_6 * cc_loss_4.item()


def compute_enc_gen_loss(model, first_gen, second_gen, first_latent, second_latent, return_grad=True):
    first_gen_disc = model.first_discriminator.forward(first_gen)
    first_gen_loss = adversarial_loss_real(first_gen_disc)

    second_gen_disc = model.second_discriminator.forward(second_gen)
    second_gen_loss = adversarial_loss_real(second_gen_disc)

    second_latent_disc = model.latent_discriminator.forward(second_latent)
    latent_loss = adversarial_loss_real(second_latent_disc)

    if return_grad:
        return W_1 * first_gen_loss, W_2 * second_gen_loss, W_l * latent_loss

    return W_1 * first_gen_loss.item(), W_2 * second_gen_loss.item(), W_l * latent_loss.item()