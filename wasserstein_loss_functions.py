

"""Wasserstein loss functions for adversarial training in LSTNET model.

This module implements Wasserstein GAN loss functions including gradient penalty,
discriminator loss computation, and generator loss computation for training
the LSTNET model with adversarial objectives across multiple domains.
"""


from typing import List, Tuple
from dataclasses import dataclass
import torch
from torch import Tensor
from models.discriminator import Discriminator
from models.lstnet import LSTNET
from utils import TensorTriplet
from loss_functions import WeightIndex


@dataclass(slots=True)
class WasserssteinTerm:
    """Represents a Wasserstein GAN loss term with gradient penalty.
    
    Attributes:
        critic_loss: The critic's loss value (Wasserstein distance estimate).
        grad_pen: The gradient penalty term for Lipschitz constraint.
        lambda_gp: Weight for the gradient penalty term (default: 10.0).
        weight: Overall weight for this loss term (default: 1.0).
    """
    critic_loss: Tensor
    grad_pen: Tensor
    lambda_gp: float = 10.0
    weight: float = 1.0

    def total_loss(self) -> Tensor:
        """Compute total Wasserstein loss including gradient penalty.
        
        Returns:
            Weighted combination of negative critic loss and scaled gradient penalty.
        """
        return self.weight * (-self.critic_loss + self.lambda_gp * self.grad_pen)

    def get_critic_loss_value(self) -> float:
        """Extract critic loss as a Python float.
        
        Returns:
            Scalar value of the critic loss.
        """
        return self.critic_loss.item()

    def get_grad_penalty_value(self) -> float:
        """Extract gradient penalty as a Python float.
        
        Returns:
            Scalar value of the gradient penalty.
        """
        return self.grad_pen.item()


def gradient_penalty(
    critic: Discriminator, batch_real: Tensor, batch_gen: Tensor
) -> Tensor:
    """Compute gradient penalty for Wasserstein GAN critic (1-Lipschitz constraint).
    
    Args:
        critic (Discriminator): The critic/discriminator network.
        batch_real (Tensor): Batch of real data samples.
        batch_gen (Tensor): Batch of generated data samples.
        
    Returns:
        Scalar gradient penalty value
    """
    batch_size = batch_real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=batch_real.device)

    # Detach generated images to prevent gradients flowing back through generator
    # Linear interpolation between real and gen imgs
    interpolation = epsilon * batch_real + (1 - epsilon) * batch_gen.detach()
    _ = interpolation.requires_grad_(True)
    critic_interpolation = critic(interpolation)

    # Compute gradients of critic outputs w.r.t. interpolated
    grads = torch.autograd.grad(
        outputs=critic_interpolation,
        inputs=interpolation,
        grad_outputs=torch.ones_like(critic_interpolation),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # grads: (B, C, H, W) → flatten per-sample
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)  # ||∇_x f(x)||_2

    # (||grad|| - 1)^2 averaged over batch
    grad_pen = ((grad_norm - 1) ** 2).mean()

    return grad_pen


def disc_wgan_loss(
    critic: Discriminator, batch_real: Tensor, batch_gen: Tensor, weight: float = 1.0
) -> WasserssteinTerm:
    """Compute Wasserstein GAN loss for discriminator.
    
    Args:
        critic (Discriminator): The critic/discriminator network.
        batch_real (Tensor): Batch of real data samples.
        batch_gen (Tensor): Batch of generated data samples.
        weight (float): Weight for this loss term (default: 1.0).
        
    Returns:
        WasserssteinTerm containing critic loss and gradient penalty.
    """
    wasserstein_term = (
        critic.forward(batch_real).mean() - critic.forward(batch_gen).mean()
    )
    grad_pen = gradient_penalty(
        critic=critic, batch_real=batch_real, batch_gen=batch_gen
    )
    return WasserssteinTerm(
        critic_loss=wasserstein_term, grad_pen=grad_pen, weight=weight
    )


def enc_gen_wgan_loss(critic: Discriminator, batch_gen: Tensor) -> Tensor:
    """Compute Wasserstein GAN loss for generator in image domain.
    
    Generator wants to maximize critic output for generated samples.
    
    Args:
        critic (Discriminator): The critic/discriminator network.
        batch_gen (Tensor): Batch of generated data samples.
        
    Returns:
        Negative mean critic output for generated samples.
    """
    """Generator wants to maximize critic output (minimize negative output)."""
    return -critic.forward(batch_gen).mean()


def enc_gen_wgan_latent_loss(
    critic: Discriminator, batch_latent_real: Tensor, batch_latent_gen: Tensor
) -> Tensor:
    """Compute Wasserstein GAN loss for generator in latent domain.
    
    Generator wants high critic scores for generated latent samples and
    low scores for real latent samples.
    
    Args:
        critic (Discriminator): The critic/discriminator network.
        batch_latent_real (Tensor): Batch of real latent samples.
        batch_latent_gen (Tensor): Batch of generated latent samples.
        
    Returns:
        Loss that maximizes critic output for generated and minimizes for real.
    """
    # high score -> real, low score -> fake
    # fool discriminator -> get high score for fake and low score for real

    # maximize: critic.forward(batch_latent_gen).mean()
    # minimize: critic.forward(batch_latent_real).mean()

    return (
        -critic.forward(batch_latent_gen).mean()
        + critic.forward(batch_latent_real).mean()
    )


def compute_discriminator_loss(
    model: LSTNET,
    weights: List[float],
    first_real_img: Tensor,
    second_real_img: Tensor,
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
) -> Tuple[WasserssteinTerm, WasserssteinTerm, WasserssteinTerm]:
    """Compute discriminator loss across all three domains.
    
    Args:
        model (LSTNET): LSTNET model containing discriminators.
        weights (List[float]): List of weights for each domain loss.
        first_real_img (Tensor): Real samples from first domain.
        second_real_img (Tensor): Real samples from second domain.
        first_gen_img (Tensor): Generated samples in first domain.
        second_gen_img (Tensor): Generated samples in second domain.
        first_latent_img (Tensor): Real latent samples from first domain.
        second_latent_img (Tensor): Generated latent samples from second domain.
        
    Returns:
        Tuple of (first_disc_loss, second_disc_loss, latent_disc_loss).
    """

    first_disc_loss = disc_wgan_loss(
        model.first_discriminator,
        first_real_img,
        first_gen_img,
        weight=weights[WeightIndex.FIRST_DOMAIN],
    )
    second_disc_loss = disc_wgan_loss(
        model.second_discriminator,
        second_real_img,
        second_gen_img,
        weight=weights[WeightIndex.SECOND_DOMAIN],
    )
    latent_disc_loss = disc_wgan_loss(
        model.latent_discriminator,
        first_latent_img,
        second_latent_img,
        weight=weights[WeightIndex.LATENT_DOMAIN],
    )

    return first_disc_loss, second_disc_loss, latent_disc_loss


def compute_enc_gen_loss(
    model: LSTNET,
    weights: List[float],
    first_gen_img: Tensor,
    second_gen_img: Tensor,
    first_latent_img: Tensor,
    second_latent_img: Tensor,
) -> TensorTriplet:
    """Compute generator/encoder loss across all three domains.
    
    Args:
        model: LSTNET model containing discriminators.
        weights: List of weights for each domain loss.
        first_gen_img: Generated samples in first domain.
        second_gen_img: Generated samples in second domain.
        first_latent_img: Real latent samples from first domain.
        second_latent_img: Generated latent samples from second domain.
        
    Returns:
        Tuple of weighted generator losses (first, second, latent).
    """

    first_gen_loss = enc_gen_wgan_loss(model.first_discriminator, first_gen_img)

    second_gen_loss = enc_gen_wgan_loss(model.second_discriminator, second_gen_img)

    latent_gen_loss = enc_gen_wgan_latent_loss(
        model.latent_discriminator, first_latent_img, second_latent_img
    )

    return (
        weights[WeightIndex.FIRST_DOMAIN] * first_gen_loss,
        weights[WeightIndex.SECOND_DOMAIN] * second_gen_loss,
        weights[WeightIndex.LATENT_DOMAIN] * latent_gen_loss,
    )
