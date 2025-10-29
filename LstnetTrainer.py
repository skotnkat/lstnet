"""
Module is implementing a trainer class for LSTNET model.
Includes training loop with optional validation (discriminator and encoder/generator updates),
early stopping based on validation loss, and optimizer setup.
"""

from typing import List, Dict, Tuple, Any, Optional
import time
import functools
import operator
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import optuna

from dual_domain_dataset import DualDomainDataset
from models.lstnet import LSTNET
import utils
from utils import FloatTriplet, FloatQuad
import loss_functions


# Constants
EXPECTED_WEIGHTS_COUNT = 7
DISCRIMINATOR_UPDATE_FREQUENCY = 2  # Update discriminator every 2nd batch


@dataclass(slots=True)
class TrainParams:
    """Class for holding training hyperparameters."""

    max_epoch_num: int = 50
    max_patience: Optional[int] = None
    optim_name: str = "Adam"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.8, 0.999)
    weight_decay: float = 0.01


class LstnetTrainer:
    """Trainer class for LSTNET model."""

    def __init__(
        self,
        lstnet_model: LSTNET,
        weights: List[float],
        train_loader: DataLoader[DualDomainDataset],
        *,
        val_loader: Optional[DataLoader[DualDomainDataset]] = None,
        train_params: TrainParams = TrainParams(),
        run_optuna: bool = False,
        optuna_trial: Optional[optuna.Trial] = None,
    ) -> None:
        """Initialize the LSTNET trainer.

        Args:
            lstnet_model (LSTNET): The LSTNET model to be trained.
            weights (List[float]): The weights for different loss components.
            train_loader (DataLoader[DualDomainDataset]): The data loader with training data.
            val_loader (Optional[DataLoader[DualDomainDataset]], optional):
                The data loader with validation data. Defaults to None.
                In case of None, no validation is performed.
            train_params (TrainParams, optional): The training hyperparameters.
            Defaults to TrainParams().

        Raises:
            ValueError: If the validation loader is not provided when validation is enabled.
            ValueError: If the number of weights is not equal to the expected count.
            ValueError: If any weight is non-positive.
            ValueError: If the learning rate is not positive.
            ValueError: If the maximum epoch number is not positive.
            ValueError: If the patience is not positive.
            ValueError: If the weight decay is negative.
        """

        self.model = lstnet_model
        self.optim_name = train_params.optim_name
        self.lr = train_params.lr
        self.betas = train_params.betas
        self.weight_decay = train_params.weight_decay

        self.disc_optim = utils.init_optimizer(
            self.optim_name,
            self.model.disc_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        self.enc_gen_optim = utils.init_optimizer(
            self.optim_name,
            self.model.enc_gen_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_validation = True if val_loader is not None else False

        if self.run_validation and val_loader is None:
            raise ValueError("Validation is enabled but no validation loader provided.")

        if len(weights) != EXPECTED_WEIGHTS_COUNT:
            raise ValueError(
                f"Missing weights. Expected {EXPECTED_WEIGHTS_COUNT} weights, \
                however only {len(weights)} provided. Please provide all required weights."
            )

        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive values.")

        if train_params.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {train_params.lr}")

        if train_params.max_epoch_num <= 0:
            raise ValueError(
                f"Max epoch number must be positive, got {train_params.max_epoch_num}"
            )

        if train_params.max_patience is not None and train_params.max_patience <= 0:
            raise ValueError(
                f"Max patience must be positive, got {train_params.max_patience}"
            )

        if not (0 <= train_params.betas[0] < 1 and 0 <= train_params.betas[1] < 1):
            raise ValueError(f"Beta values must be in [0, 1), got {train_params.betas}")

        if train_params.weight_decay < 0:
            raise ValueError(
                f"Weight decay must be non-negative, got {train_params.weight_decay}"
            )

        self.weights = weights

        # Adjust max_patience if not provided (None) -> set to max_epoch_num to never early stop
        self.max_patience = train_params.max_patience
        if train_params.max_patience is None:
            self.max_patience = train_params.max_epoch_num

        self.max_epoch_num = train_params.max_epoch_num

        self.best_state_dict = self.model.state_dict()
        self.best_loss = np.inf
        self.best_epoch_idx = None
        self.cur_patience = 0
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []

        self.run_optuna = run_optuna
        self.optuna_trial = optuna_trial

    def get_trainer_info(self) -> Dict[str, Any]:
        """
        Function to get all the relevant trainer information.
        Used for logging and analysis.
        """

        return {
            "train_loss": self.train_loss_list,
            "val_loss": self.val_loss_list,
            "epoch_num": len(self.train_loss_list),
            "best_epoch": self.best_epoch_idx,
            "best_loss": self.best_loss,
            "max_patience": self.max_patience,
            "max_epoch_num": self.max_epoch_num,
            "weights": self.weights,
            "optimizer": self.optim_name,
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "run_optuna": self.run_optuna,
        }

    def _update_disc(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.disc_optim.zero_grad()

        with torch.no_grad():
            # second_gen_img are images from first domain mapped to second domain
            second_gen_img, first_latent_img = self.model.map_first_to_second(
                first_real_img, return_latent=True
            )

            # first_gen_img are images from the second domain mapped to first domain
            first_gen_img, second_latent_img = self.model.map_second_to_first(
                second_real_img, return_latent=True
            )
            imgs_mapping = (
                first_gen_img,
                second_gen_img,
                first_latent_img,
                second_latent_img,
            )

        disc_loss_tensors = loss_functions.compute_discriminator_loss(
            self.model,
            self.weights,
            first_real_img,
            second_real_img,
            *imgs_mapping,
        )

        total_disc_loss = functools.reduce(operator.add, disc_loss_tensors)

        # only for obtaining all the losses, no update
        with torch.no_grad():
            imgs_cc = self.model.get_cc_components(*imgs_mapping)
            cc_loss_tuple = loss_functions.compute_cc_loss(
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_cc,
                return_grad=False,
            )
            enc_gen_loss_tuple = loss_functions.compute_enc_gen_loss(
                self.model, self.weights, *imgs_mapping, return_grad=False
            )

        total_disc_loss.backward()
        self.disc_optim.step()

        d1, d2, d3 = disc_loss_tensors
        disc_loss_tuple = (d1.item(), d2.item(), d3.item())

        return disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple

    def _update_enc_gen(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.enc_gen_optim.zero_grad()

        # Generate images from first domain to second and vice versa
        second_gen_img, first_latent_img = self.model.map_first_to_second(
            first_real_img, return_latent=True
        )
        first_gen_img, second_latent_img = self.model.map_second_to_first(
            second_real_img, return_latent=True
        )
        imgs_mapping = (
            first_gen_img,
            second_gen_img,
            first_latent_img,
            second_latent_img,
        )
        imgs_cc = self.model.get_cc_components(*imgs_mapping)

        cc_loss_tensors = loss_functions.compute_cc_loss(
            self.weights, first_real_img, second_real_img, *imgs_cc
        )
        enc_gen_loss_tensors = loss_functions.compute_enc_gen_loss(
            self.model, self.weights, *imgs_mapping
        )

        total_enc_gen_loss = functools.reduce(
            operator.add, cc_loss_tensors
        ) + functools.reduce(operator.add, enc_gen_loss_tensors)

        # only for obtaining all losses, no update
        with torch.no_grad():
            disc_loss_tuple = loss_functions.compute_discriminator_loss(
                self.model,
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_mapping,
                return_grad=False,
            )

        total_enc_gen_loss.backward()
        self.enc_gen_optim.step()

        cc1, cc2, cc3, cc4 = cc_loss_tensors
        cc_loss_tuple = (cc1.item(), cc2.item(), cc3.item(), cc4.item())

        eg1, eg2, eg3 = enc_gen_loss_tensors
        enc_gen_loss_tuple = (eg1.item(), eg2.item(), eg3.item())

        return disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple

    def _run_eval_loop(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        with torch.no_grad():
            # Generate images from first domain to second and vice versa
            second_gen_img, first_latent_img = self.model.map_first_to_second(
                first_real_img, return_latent=True
            )
            first_gen_img, second_latent_img = self.model.map_second_to_first(
                second_real_img, return_latent=True
            )
            imgs_mapping = (
                first_gen_img,
                second_gen_img,
                first_latent_img,
                second_latent_img,
            )
            imgs_cc = self.model.get_cc_components(*imgs_mapping)

            disc_loss_tuple = loss_functions.compute_discriminator_loss(
                self.model,
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_mapping,
                return_grad=False,
            )
            cc_loss_tuple = loss_functions.compute_cc_loss(
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_cc,
                return_grad=False,
            )
            enc_gen_loss_tuple = loss_functions.compute_enc_gen_loss(
                self.model, self.weights, *imgs_mapping, return_grad=False
            )

        return disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple

    def _fit_epoch(self) -> float:
        epoch_loss = 0

        for batch_idx, (first_real, _, second_real, _) in enumerate(self.train_loader):
            first_real = first_real.to(utils.DEVICE)
            second_real = second_real.to(utils.DEVICE)

            if batch_idx % DISCRIMINATOR_UPDATE_FREQUENCY == 0:
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = self._update_disc(
                    first_real, second_real
                )
            else:
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = (
                    self._update_enc_gen(first_real, second_real)
                )

            epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

            utils.log_epoch_loss(
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple, "train"
            )

        scale = len(self.train_loader)
        utils.normalize_epoch_loss(scale, "train")
        epoch_loss /= scale

        return epoch_loss

    def _run_eval_epoch(self) -> float:
        epoch_loss = 0

        # Type assertion since we validate val_loader is not None in __init__
        assert self.val_loader is not None, "Validation loader should be available"

        for _, (first_real, _, second_real, _) in enumerate(self.val_loader):
            first_real_img = first_real.to(utils.DEVICE)
            second_real_img = second_real.to(utils.DEVICE)

            disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = self._run_eval_loop(
                first_real_img, second_real_img
            )

            epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

            utils.log_epoch_loss(
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple, "val"
            )

        scale = len(self.val_loader)
        utils.normalize_epoch_loss(scale, "val")
        epoch_loss /= scale

        return epoch_loss

    def _run_epoch(self, val_op: bool = False) -> float:
        if val_op:
            if self.val_loader is None:
                raise ValueError("No validation loader provided.")

            return self._run_eval_epoch()

        return self._fit_epoch()

    def fit(self) -> LSTNET:
        """Fit the LSTNET model.

        Returns:
            LSTNET: The trained LSTNET model.
        """
        _ = self.model.to(utils.DEVICE)

        epoch_idx = 0  # Init outside loop scope
        for epoch_idx in range(self.max_epoch_num):
            start_time = time.time()
            utils.init_epoch_loss(op="train")
            epoch_loss = self._run_epoch(val_op=False)
            self.train_loss_list.append(epoch_loss)

            if self.run_validation:
                utils.init_epoch_loss(op="val")
                epoch_loss = self._run_epoch(val_op=True)
                self.val_loss_list.append(epoch_loss)

                # ------------------------------
                # Pruning in case of optuna
                if self.run_optuna and self.optuna_trial is None:
                    raise ValueError(
                        "Optuna trial is None, but run_optuna is set to True. Provide valid optuna trial."
                    )

                self.optuna_trial.report(epoch_loss, epoch_idx)
                if self.optuna_trial.should_prune():
                    self.optuna_trial.set_user_attr("train_logs", utils.LOSS_LOGS.copy())
                    raise optuna.TrialPruned()

                # ------------------------------

            if epoch_loss < self.best_loss:
                self.best_state_dict = self.model.state_dict()
                self.best_loss = epoch_loss
                self.best_epoch_idx = epoch_idx
                self.cur_patience = 0

            else:
                self.cur_patience += 1

                if self.cur_patience >= self.max_patience:  # type: ignore
                    print("Max patience reached")
                    break

            end_time = time.time()

            print(f"\tEpoch {epoch_idx} took: {(end_time - start_time) / 60:.2f} min")
            print(f"\tPatience: {self.cur_patience}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _ = self.model.to("cpu")

        print(f"Best model reached in epoch: {self.best_epoch_idx}")
        print(f"Training ended after: {epoch_idx+1} epochs")

        _ = self.model.load_state_dict(self.best_state_dict)

        return self.model
