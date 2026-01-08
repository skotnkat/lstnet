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
from utils import TensorQuad, FloatTriplet, FloatQuad
import loss_functions
import wasserstein_loss_functions
from wasserstein_loss_functions import WasserssteinTerm


# Constants
EXPECTED_WEIGHTS_COUNT = 7


@dataclass(slots=True)
class TrainParams:
    """Class for holding training hyperparameters."""

    max_epoch_num: int = 50
    max_patience: Optional[int] = None
    optim_name: str = "Adam"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.8, 0.999)
    weight_decay: float = 0.01
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6


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
        disc_update_freq: int = 2,  
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

        # Initialize learning rate schedulers
        self.disc_scheduler = None
        self.enc_gen_scheduler = None

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
        self.wasserstein = False

        # Adjust max_patience if not provided (None) -> set to max_epoch_num to never early stop
        self.max_patience = train_params.max_patience
        if train_params.max_patience is None:
            self.max_patience = train_params.max_epoch_num

        self.max_epoch_num = train_params.max_epoch_num
        self.disc_update_freq = disc_update_freq

        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []

        self.best_val_loss = np.inf
        self.patience_counter = 0

        self.run_optuna = run_optuna
        self.optuna_trial = optuna_trial
        self.fin_loss = np.inf

        self.loss_types = ["disc_loss", "enc_gen_loss", "cc_loss"]
        self.loss_logs: Dict[str, Dict[str, Dict[str, List[float]]]] = dict()


    def get_trainer_info(self) -> Dict[str, Any]:
        """
        Function to get all the relevant trainer information.
        Used for logging and analysis.
        """

        return {
            "train_loss": self.train_loss_list,
            "val_loss": self.val_loss_list,
            "fin_loss": self.fin_loss,
            "epoch_num": len(self.train_loss_list),
            "max_patience": self.max_patience,
            "max_epoch_num": self.max_epoch_num,
            "weights": self.weights,
            "optimizer": self.optim_name,
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "run_optuna": self.run_optuna,
        }

    def get_trans_imgs(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> TensorQuad:
        """Generate translated images from both domains.

        Args:
            first_real_img (Tensor): Real images from the first domain.
            second_real_img (Tensor): Real images from the second domain.
        """
        second_gen_img, first_latent_img = self.model.map_first_to_second(
            first_real_img
        )

        # first_gen_img are images from the second domain mapped to first domain
        first_gen_img, second_latent_img = self.model.map_second_to_first(
            second_real_img
        )

        imgs_mapping = (
            first_gen_img,
            second_gen_img,
            first_latent_img,
            second_latent_img,
        )

        return imgs_mapping

    def _get_disc_losses(
        self,
        first_real_img: Tensor,
        second_real_img: Tensor,
        imgs_mapping: TensorQuad,
        compute_grad: bool = True,
    ) -> Any:
        loss_func = loss_functions.compute_discriminator_loss
        if self.wasserstein:
            loss_func = wasserstein_loss_functions.compute_discriminator_loss

        if compute_grad:
            return loss_func(
                self.model,
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_mapping,
            )

        else:
            imgs_mapping_detached = (img.detach() for img in imgs_mapping)
            disc_loss_tensors = loss_func(
                self.model,
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_mapping_detached,
            )

        return disc_loss_tensors

    def _get_enc_gen_losses(
        self, imgs_mapping: TensorQuad, compute_grad: bool = True
    ) -> Any:
        loss_func = loss_functions.compute_enc_gen_loss
        if self.wasserstein:
            loss_func = wasserstein_loss_functions.compute_enc_gen_loss
        if compute_grad:
            return loss_func(
                self.model,
                self.weights,
                *imgs_mapping,
            )
        else:
            with torch.no_grad():
                enc_gen_loss_tensors = loss_func(
                    self.model,
                    self.weights,
                    *imgs_mapping,
                )
        return enc_gen_loss_tensors

    def _get_cc_losses(
        self,
        first_real_img: Tensor,
        second_real_img: Tensor,
        imgs_mapping: TensorQuad,
        compute_grad: bool = True,
    ) -> Any:
        imgs_cc = self.model.get_cc_components(*imgs_mapping)

        if compute_grad:
            return loss_functions.compute_cc_loss(
                self.weights,
                first_real_img,
                second_real_img,
                *imgs_cc,
            )
        else:
            with torch.no_grad():
                cc_loss_tensors = loss_functions.compute_cc_loss(
                    self.weights,
                    first_real_img,
                    second_real_img,
                    *imgs_cc,
                )
        return cc_loss_tensors

    def _get_losses(
        self, first_real_img: Tensor, second_real_img: Tensor, op: str = "eval"
    ) -> Any:  # TBD
        if op not in ["disc", "enc_gen", "eval"]:
            raise ValueError(
                f"Invalid operation '{op}'. Expected 'disc', 'enc_gen', or 'eval'."
            )

        imgs_mapping = self.get_trans_imgs(first_real_img, second_real_img)

        disc_grad, enc_gen_grad, cc_grad = False, False, False  # for eval
        if op == "disc":
            disc_grad = True
        elif op == "enc_gen":
            enc_gen_grad = True
            cc_grad = True

        disc_loss = self._get_disc_losses(
            first_real_img, second_real_img, imgs_mapping, compute_grad=disc_grad
        )
        enc_gen_loss = self._get_enc_gen_losses(imgs_mapping, compute_grad=enc_gen_grad)
        cc_loss = self._get_cc_losses(
            first_real_img, second_real_img, imgs_mapping, compute_grad=cc_grad
        )

        return disc_loss, enc_gen_loss, cc_loss

    @staticmethod
    def _convert_losses_to_floats(losses: List[Any]) -> List[Any]:
        return [utils.convert_tensor_tuple_to_floats(loss) for loss in losses]

    def _update_disc(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Any:  # Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.disc_optim.zero_grad()

        disc_loss, enc_gen_loss, cc_loss = self._get_losses(
            first_real_img, second_real_img, op="disc"
        )

        total_disc_loss = functools.reduce(operator.add, disc_loss)
        total_disc_loss.backward()
        self.disc_optim.step()

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])

    def _update_enc_gen(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Any:  # Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.enc_gen_optim.zero_grad()

        disc_loss, enc_gen_loss, cc_loss = self._get_losses(
            first_real_img, second_real_img, op="enc_gen"
        )

        total_enc_gen_loss = functools.reduce(operator.add, cc_loss) + functools.reduce(
            operator.add, enc_gen_loss
        )

        total_enc_gen_loss.backward()
        self.enc_gen_optim.step()

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])


    def eval_forward(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Any:  # Tuple[TensorTriplet, TensorTriplet, TensorQuad]
        # Generate images from first domain to second and vice versa
        second_gen_img, first_latent_img = self.model.map_first_to_second(
            first_real_img
        )
        first_gen_img, second_latent_img = self.model.map_second_to_first(
            second_real_img
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


    def _run_eval_loop(self, first_real_img: Tensor, second_real_img: Tensor) -> Any:
        self.model.eval()
        
        with torch.no_grad():
            disc_loss, enc_gen_loss, cc_loss = self._get_losses(
                first_real_img, second_real_img, op="eval"
            )

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])

    def _fit_epoch(self) -> float:
        epoch_loss = 0

        for batch_idx, (first_real, _, second_real, _) in enumerate(self.train_loader):
            first_real = first_real.to(utils.DEVICE)
            second_real = second_real.to(utils.DEVICE)

            if batch_idx % self.disc_update_freq == 0:
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = (
                    self._update_enc_gen(first_real, second_real)
                )

            else:
                disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple = self._update_disc(
                    first_real, second_real
                )


            epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

            self.log_epoch_loss(
                [disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple], "train"
            )

        scale = len(self.train_loader)
        self.normalize_epoch_loss(scale, "train")
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

            self.log_epoch_loss(
                [disc_loss_tuple, enc_gen_loss_tuple, cc_loss_tuple],
                "val",
            )

        scale = len(self.val_loader)
        self.normalize_epoch_loss(scale, "val")
        epoch_loss /= scale

        return epoch_loss

    def _run_epoch(self, val_op: bool = False) -> float:
        if val_op:
            if self.val_loader is None:
                raise ValueError("No validation loader provided.")

            return self._run_eval_epoch()

        self.model.train()
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
            self.init_epoch_loss(op="train")
            epoch_loss = self._run_epoch(val_op=False)
            self.train_loss_list.append(epoch_loss)
            
            # Reshuffle second dataset indices for next epoch to avoid consistent pairs of images
            self.train_loader.dataset._shuffle_second_indices()

            if self.run_validation:
                self.init_epoch_loss(op="val")
                epoch_loss = self._run_epoch(val_op=True)
                self.val_loss_list.append(epoch_loss)
                
                # ------------------------------
                # Early stopping check
                if epoch_loss < self.best_val_loss:
                    self.best_val_loss = epoch_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.max_patience:
                    if not self.run_optuna:
                        print(
                            f"\nEarly stopping triggered at epoch {epoch_idx}. Patience limit reached."
                        )
                    break

                # ------------------------------
                # Pruning in case of optuna
                if self.run_optuna and self.optuna_trial is None:
                    raise ValueError(
                        "Optuna trial is None, but run_optuna is set to True. Provide valid optuna trial."
                    )

                if self.run_optuna:
                    self.optuna_trial.report(epoch_loss, epoch_idx)
                    if self.optuna_trial.should_prune():
                        self.optuna_trial.set_user_attr(
                            "train_logs", self.loss_logs.copy()
                        )
                        raise optuna.TrialPruned()

                # ------------------------------

            end_time = time.time()

            if not self.run_optuna:
                print(
                    f"\tEpoch {epoch_idx} took: {(end_time - start_time) / 60:.2f} min"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            

        _ = self.model.to("cpu")
        self.fin_loss = (
            self.val_loss_list[-1] if self.run_validation else self.train_loss_list[-1]
        )

        return self.model

    def init_logs(self, ops: Optional[List[str]] = None) -> None:
        """Initialize logs for training and validation operations."""

        if ops is None:
            ops = ["train", "val"]

        for op in ops:
            self.loss_logs[op] = dict()
            for key in self.loss_types:
                self.loss_logs[op][key] = dict()
                values = ["first", "second", "latent"]
                if key == "cc_loss":
                    values = [
                        "first_cycle",
                        "second_cycle",
                        "first_full_cycle",
                        "second_full_cycle",
                    ]

                for val in values:
                    self.loss_logs[op][key][f"{val}_loss"] = []

    def init_epoch_loss(self, op: str) -> None:
        for loss_type in self.loss_types:
            values = ["first", "second", "latent"]
            if loss_type == "cc_loss":
                values = [
                    "first_cycle",
                    "second_cycle",
                    "first_full_cycle",
                    "second_full_cycle",
                ]
            for val in values:
                self.loss_logs[op][loss_type][f"{val}_loss"].append(0.0)

    def log_epoch_loss(self, loss_values: List[Any], op: str):
        """
        Log epoch loss for a given operation.

        Args:
            disc_loss (FloatTriplet): Discriminator loss.
                Consist of first, second and latent loss.
            enc_gen_loss (FloatTriplet): Encoder-Generator loss.
                Consists of first, second and latent loss.
            cc_loss (FloatQuad): Cycle consistency loss.
            first_cycle, second_cycle, first_full_cycle, second_full_cycle).
                Consists of first_cycle, second_cycle, first_full_cycle, second_full_cycle.
            op (str): Operation type.
        """

        op_logs = self.loss_logs[op]
        cur_epoch = len(op_logs[self.loss_types[0]]["first_loss"]) - 1  # last epoch

        for loss_type, loss_value in zip(self.loss_types, loss_values):
            if loss_type != "cc_loss":
                op_logs[loss_type]["first_loss"][cur_epoch] += loss_value[0]
                op_logs[loss_type]["second_loss"][cur_epoch] += loss_value[1]
                op_logs[loss_type]["latent_loss"][cur_epoch] += loss_value[2]

            else:
                op_logs[loss_type]["first_cycle_loss"][cur_epoch] += loss_value[0]
                op_logs[loss_type]["second_cycle_loss"][cur_epoch] += loss_value[1]
                op_logs[loss_type]["first_full_cycle_loss"][cur_epoch] += loss_value[2]
                op_logs[loss_type]["second_full_cycle_loss"][cur_epoch] += loss_value[3]

    def normalize_epoch_loss(self, scale, op) -> None:
        """
        Normalize epoch loss for a given operation.

        Args:
            scale (float): Scaling factor.
            op (str): Operation type.
        """

        op_logs = self.loss_logs[op]
        cur_epoch = len(op_logs["disc_loss"]["first_loss"]) - 1  # last epoch

        for loss_type in op_logs.keys():
            for key in op_logs[loss_type].keys():
                op_logs[loss_type][key][cur_epoch] /= scale


class WassersteinLstnetTrainer(LstnetTrainer):
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
        disc_update_freq: int = 2
    ) -> None:
        super().__init__(
            lstnet_model,
            weights,
            train_loader,
            val_loader=val_loader,
            train_params=train_params,
            run_optuna=run_optuna,
            optuna_trial=optuna_trial,
        )

        self.wasserstein = True
        self.loss_types = ["disc_loss", "grad_pen", "enc_gen_loss", "cc_loss"]

    @staticmethod
    def _convert_losses_to_floats(losses: List[Any]) -> List[Any]:
        disc_loss, enc_gen_loss, cc_loss = losses

        if isinstance(disc_loss[0], WasserssteinTerm):
            disc_losses_values = tuple(
                loss.get_critic_loss_value() for loss in disc_loss
            )
            grad_penalties = tuple(loss.get_grad_penalty_value() for loss in disc_loss)
        else:
            disc_losses_values = utils.convert_tensor_tuple_to_floats(disc_loss)
            grad_penalties = (0.0, 0.0, 0.0)

        enc_gen_loss_values = utils.convert_tensor_tuple_to_floats(enc_gen_loss)
        cc_loss_values = utils.convert_tensor_tuple_to_floats(cc_loss)

        return disc_losses_values, grad_penalties, enc_gen_loss_values, cc_loss_values

    def _update_disc(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Any:  # Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.disc_optim.zero_grad()

        disc_loss, enc_gen_loss, cc_loss = self._get_losses(
            first_real_img, second_real_img, op="disc"
        )
        disc_losses_all = [loss.total_loss() for loss in disc_loss]

        total_disc_loss = functools.reduce(operator.add, disc_losses_all)
        total_disc_loss.backward()
        self.disc_optim.step()

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])

    def _update_enc_gen(
        self, first_real_img: Tensor, second_real_img: Tensor
    ) -> Any:  # Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        self.enc_gen_optim.zero_grad()

        disc_loss, enc_gen_loss, cc_loss = self._get_losses(
            first_real_img, second_real_img, op="enc_gen"
        )

        total_enc_gen_loss = functools.reduce(operator.add, cc_loss) + functools.reduce(
            operator.add, enc_gen_loss
        )

        total_enc_gen_loss.backward()
        self.enc_gen_optim.step()

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])

    def _run_eval_loop(
        self, first_real_img: Tensor, second_real_img: Tensor
        ) -> Tuple[FloatTriplet, FloatTriplet, FloatQuad]:
        disc_loss, enc_gen_loss, cc_loss = self._get_losses(
            first_real_img, second_real_img, op="eval"
        )

        return self._convert_losses_to_floats([disc_loss, enc_gen_loss, cc_loss])

    def _fit_epoch(self) -> float:
        epoch_loss = 0

        for batch_idx, (first_real, _, second_real, _) in enumerate(self.train_loader):
            first_real = first_real.to(utils.DEVICE)
            second_real = second_real.to(utils.DEVICE)

            if batch_idx % self.disc_update_freq == 0:
                disc_loss_tuple, grad_pen_tuple, enc_gen_loss_tuple, cc_loss_tuple = (
                    self._update_enc_gen(first_real, second_real)
                )

            else:
                disc_loss_tuple, grad_pen_tuple, enc_gen_loss_tuple, cc_loss_tuple = (
                    self._update_disc(first_real, second_real)
                )

            epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

            self.log_epoch_loss(
                [disc_loss_tuple, grad_pen_tuple, enc_gen_loss_tuple, cc_loss_tuple],
                "train",
            )

        scale = len(self.train_loader)
        self.normalize_epoch_loss(scale, "train")
        epoch_loss /= scale

        return epoch_loss

    def _run_eval_epoch(self) -> float:
        epoch_loss = 0

        # Type assertion since we validate val_loader is not None in __init__
        assert self.val_loader is not None, "Validation loader should be available"

        for _, (first_real, _, second_real, _) in enumerate(self.val_loader):
            first_real_img = first_real.to(utils.DEVICE)
            second_real_img = second_real.to(utils.DEVICE)

            disc_loss_tuple, grad_pen_tuple, enc_gen_loss_tuple, cc_loss_tuple = (
                self._run_eval_loop(first_real_img, second_real_img)
            )

            epoch_loss += sum(disc_loss_tuple) + sum(cc_loss_tuple)

            self.log_epoch_loss(
                [disc_loss_tuple, grad_pen_tuple, enc_gen_loss_tuple, cc_loss_tuple],
                "val",
            )

        scale = len(self.val_loader)
        self.normalize_epoch_loss(scale, "val")
        epoch_loss /= scale

        return epoch_loss
