import numpy as np
from loss_functions import compute_discriminator_loss, compute_enc_gen_loss


def compute_loss(model, first_domain_batch, second_domain_batch):
    total_disc_loss = compute_discriminator_loss(model, first_domain_batch, second_domain_batch)
    total_enc_gen_loss = compute_enc_gen_loss(model, first_domain_batch, second_domain_batch)

    return total_disc_loss, total_enc_gen_loss
