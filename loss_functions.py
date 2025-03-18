import torch.nn as nn
import torch
import utils

W1, W2, W3, W4, W5, W6, W_l = utils.initialize_weights()
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



