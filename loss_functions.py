"""
Module is implementing loss functions used for training of encoder-generator-discriminator network.
To be specific, loss functions needed for training o LSTNET model.
"""

from typing import List, Union, overload, Literal
from enum import IntEnum
import torch.nn as nn
import torch
from torch import Tensor

from models.lstnet import LSTNET
from utils import (
    TensorTriplet,
    TensorQuad,
    FloatTriplet,
    FloatQuad,
    convert_tensor_tuple_to_floats,
)

import wasserstein_loss_functions


class WeightIndex(IntEnum):
    """Indices for the weights list used in loss computations."""

    FIRST_DOMAIN = 0
    SECOND_DOMAIN = 1
    LATENT_DOMAIN = 2
    CC_FIRST_DOMAIN = 3
    CC_SECOND_DOMAIN = 4
    CC_FULL_FIRST_DOMAIN = 5
    CC_FULL_SECOND_DOMAIN = 6


adversarial_loss = nn.BCEWithLogitsLoss()
cycle_loss = nn.L1Loss()


def adversarial_loss_real(batch: Tensor) -> Tensor:
    """
    Computes the adversarial loss for real images.
    Their labels are expected to be ones.

    Args:
        batch (Tensor): The batch of real images.

    Returns:
        Tensor: The computed adversarial loss.
    """

    batch_size = batch.size(0)
    ones_labels = torch.ones(
        batch_size, 1, device=batch.device, requires_grad=False
    )  # expecting to be real

    return adversarial_loss(batch, ones_labels)


def adversarial_loss_fake(batch: Tensor) -> Tensor:
    """
    Computes the adversarial loss for fake images.
    Their labels are expected to be zeros.

    Args:
        batch (Tensor): The batch of fake images.

    Returns:
        Tensor: The computed adversarial loss.
    """

    batch_size = batch.size(0)
    zeros_labels = torch.zeros(
        batch_size, 1, device=batch.device, requires_grad=False
    )  # expecting to be fake

    return adversarial_loss(batch, zeros_labels)


def network_adversarial_loss(batch_real: Tensor, batch_gen: Tensor) -> Tensor:
    """
    Computes the adversarial loss for the network.
    The loss is a sum of the losses for real and generated images.
    The real images are expected to be real (labels are ones).
    The generated images are expected to be fake (labels are zeros).

    Args:
        batch_real (Tensor): The batch of real images.
        batch_gen (Tensor): The batch of generated images.

    Returns:
        Tensor: The computed adversarial loss.
    """

    real_loss = adversarial_loss_real(batch_real)
    gen_loss = adversarial_loss_fake(batch_gen)

    return real_loss + gen_loss


# Overloads for type checking
@overload
def compute_discriminator_loss(
    model: LSTNET,
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
) -> TensorTriplet: ...
@overload
def compute_discriminator_loss(
    model: LSTNET,
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: Literal[True],
) -> TensorTriplet: ...
@overload
def compute_discriminator_loss(
    model: LSTNET,
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: Literal[False],
) -> FloatTriplet: ...


def compute_discriminator_loss(
    model: LSTNET,
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: bool = True,
) -> Union[TensorTriplet, FloatTriplet]:
    """
    Computes the discriminator loss for the given images.
    Runs the images through the respective discriminators of the model
    and computes the loss for each domain.

    The first discriminator is expected to label the first real images as ones,
    and the first generated images as zeros. The same for the second discriminator.

    The latent discriminator is expected to label the latent images from first domain as ones,
    and the latent images from second domain as zeros.

    Args:
        model (LSTNET): The LSTNET model.
        weights (List[float]): The weights for each domain.
        first_real_img (Tensor): The first real images.
        second_real_img (Tensor): The second real images.
        first_gen_img (Tensor): Images from second domain mapped to the first domain.
        second_gen_img (Tensor): Images from first domain mapped to the second domain.
        first_latent_img (Tensor): Images from first domain mapped to latent.
        second_latent_img (Tensor): Images from second domain mapped to latent.
        return_grad (bool, optional): Whether to return gradients.
        Defaults to True.

    Returns:
        Union[TensorTriplet, FloatTriplet]: The computed discriminator loss.
    """
    if wasserstein:
        first_disc_loss, second_disc_loss, latent_disc_loss = (
            wasserstein_loss_functions.compute_discriminator_loss(
                model,
                first_real_img,
                second_real_img,
                first_gen_img,
                second_gen_img,
                first_latent_img,
                second_latent_img,
            )
        )

    else:
        first_real_disc = model.first_discriminator.forward(first_real_img)
        first_gen_disc = model.first_discriminator.forward(first_gen_img)
        first_disc_loss = network_adversarial_loss(first_real_disc, first_gen_disc)

        second_real_disc = model.second_discriminator.forward(second_real_img)
        second_gen_disc = model.second_discriminator.forward(second_gen_img)
        second_disc_loss = network_adversarial_loss(second_real_disc, second_gen_disc)

        first_latent_disc = model.latent_discriminator.forward(first_latent_img)
        second_latent_disc = model.latent_discriminator.forward(second_latent_img)
        latent_disc_loss = network_adversarial_loss(
            first_latent_disc, second_latent_disc
        )

    res_with_grad = (
        weights[WeightIndex.FIRST_DOMAIN] * first_disc_loss,
        weights[WeightIndex.SECOND_DOMAIN] * second_disc_loss,
        weights[WeightIndex.LATENT_DOMAIN] * latent_disc_loss,
    )
    if return_grad:
        return res_with_grad

    return convert_tensor_tuple_to_floats(res_with_grad)


# Overloads for type checking
@overload
def compute_cc_loss(
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_cycle_img: Tensor,
    second_cycle_img: Tensor,
    first_full_cycle_img: Tensor,
    second_full_cycle_img: Tensor,
) -> TensorQuad: ...
@overload
def compute_cc_loss(
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_cycle_img: Tensor,
    second_cycle_img: Tensor,
    first_full_cycle_img: Tensor,
    second_full_cycle_img: Tensor,
    return_grad: Literal[True],
) -> TensorQuad: ...
@overload
def compute_cc_loss(
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_cycle_img: Tensor,
    second_cycle_img: Tensor,
    first_full_cycle_img: Tensor,
    second_full_cycle_img: Tensor,
    return_grad: Literal[False],
) -> FloatQuad: ...
def compute_cc_loss(
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_cycle_img: Tensor,
    second_cycle_img: Tensor,
    first_full_cycle_img: Tensor,
    second_full_cycle_img: Tensor,
    return_grad: bool = True,
) -> Union[TensorQuad, FloatQuad]:
    """
    Computes the cycle-consistency loss for the given images.
    It is computed to improve the common latent space representation.

    The loss is computed as the L1 loss between the real images and the cycle images
    (after half or full cycle).

    By half-cycle image from the first domain, we mean an image from first domain
    was translated to the common latent space and then back to the first domain.
    The same applies to the second domain.

    By full-cycle image from the first domain, we mean an image from first domain
    that was translated to the second domain and then back to the first domain.

    Args:
        weights (List[float]): _description_
        first_real_img (Tensor): _description_
        second_real_img (Tensor): _description_
        first_cycle_img (Tensor): _description_
        second_cycle_img (Tensor): _description_
        first_full_cycle_img (Tensor): _description_
        second_full_cycle_img (Tensor): _description_
        return_grad (bool, optional): _description_. Defaults to True.

    Returns:
        Union[TensorQuad, FloatQuad]: _description_
    """
    cc_loss_1 = cycle_loss(first_cycle_img, first_real_img)  # (predictions, target)
    cc_loss_2 = cycle_loss(second_cycle_img, second_real_img)

    cc_loss_3 = cycle_loss(first_full_cycle_img, first_real_img)
    cc_loss_4 = cycle_loss(second_full_cycle_img, second_real_img)

    res_with_grad = (
        weights[WeightIndex.CC_FIRST_DOMAIN] * cc_loss_1,
        weights[WeightIndex.CC_SECOND_DOMAIN] * cc_loss_2,
        weights[WeightIndex.CC_FULL_FIRST_DOMAIN] * cc_loss_3,
        weights[WeightIndex.CC_FULL_SECOND_DOMAIN] * cc_loss_4,
    )
    if return_grad:
        return res_with_grad

    return convert_tensor_tuple_to_floats(res_with_grad)


# Overloads for type checking
@overload
def compute_enc_gen_loss(
    model: LSTNET,
    weights: List[float],
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
) -> TensorTriplet: ...
@overload
def compute_enc_gen_loss(
    model: LSTNET,
    weights: List[float],
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: Literal[True],
) -> TensorTriplet: ...
@overload
def compute_enc_gen_loss(
    model: LSTNET,
    weights: List[float],
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: Literal[False],
) -> FloatTriplet: ...


def compute_enc_gen_loss(
    model: LSTNET,
    weights: List[float],
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
    wasserstein: bool,
    return_grad: bool = True,
) -> Union[TensorTriplet, FloatTriplet]:
    """
    Computes the encoder-generator loss for the given images.

    The goal of encoder-generator components is to fool the discriminators
    by producing realistic images (mapping them realistically from one domain to the other).

    The loss is computed as the adversarial loss for the generated images.
    The generated images from first domain are expected to be labeled by discriminator as real,
    therefore labeled as second. The same applies to the second domain.

    For the latent images, the goal is to create common latent space representation,
    where the latent images from first domain are indistinguishable from the ones
    from the second domain. Therefore, the latent images from first domain
    are expected to be labeled as the ones from the second domain and vice versa.

    Args:
        model (LSTNET): _description_
        weights (List[float]): _description_
        first_gen_img (Tensor): _description_
        second_gen_img (Tensor): _description_
        first_latent_img (Tensor): _description_
        second_latent_img (Tensor): _description_
        return_grad (bool, optional): _description_. Defaults to True.

    Returns:
        Union[TensorTriplet, FloatTriplet]: _description_
    """

    if wasserstein:
        first_gen_loss, second_gen_loss, latent_loss = (
            wasserstein_loss_functions.compute_enc_gen_loss(
                model,
                first_gen_img,
                second_gen_img,
                first_latent_img,
                second_latent_img,
            )
        )

    else:
        first_gen_disc = model.first_discriminator.forward(first_gen_img)
        first_gen_loss = adversarial_loss_real(first_gen_disc)

        second_gen_disc = model.second_discriminator.forward(second_gen_img)
        second_gen_loss = adversarial_loss_real(second_gen_disc)

        first_latent_disc = model.latent_discriminator.forward(first_latent_img)
        second_latent_disc = model.latent_discriminator.forward(second_latent_img)
        latent_loss = network_adversarial_loss(second_latent_disc, first_latent_disc)

    res_with_grad = (
        weights[WeightIndex.FIRST_DOMAIN] * first_gen_loss,
        weights[WeightIndex.SECOND_DOMAIN] * second_gen_loss,
        weights[WeightIndex.LATENT_DOMAIN] * latent_loss,
    )
    if return_grad:
        return res_with_grad

    return convert_tensor_tuple_to_floats(res_with_grad)
