"""Classifier models and their training."""

from typing import Tuple, Sequence, Any, Optional, List, Dict
import time
import numpy as np
import copy

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

import optuna

from models.discriminator import Discriminator
import utils
from models.extended_layers import Conv2dExtended, MaxPool2dExtended
from models.resnet_18 import ResNet18


class BaseClf(Discriminator):
    """
    Base classifier model. Inherits from Discriminator.
    
    Args:
        input_size (Tuple[int, int]): Input image size (height, width).
        in_channels (int): Number of input channels.
        params (Sequence[Any]): Model architecture parameters.
        use_checkpoint (bool, optional): Whether to use gradient checkpointing for
            memory efficiency during training. Defaults to False.
    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels: int,
        params: Sequence[Any],
        use_checkpoint: bool = False,
    ):
        self.input_size = input_size
        self.in_channels_num = in_channels

        super().__init__(
            self.input_size,
            self.in_channels_num,
            params[:-1],
            negative_slope=params[-1]["leaky_relu_neg_slope"],
            use_checkpoint=use_checkpoint,
        )

    def _create_last_layer(self) -> nn.Sequential:
        """
        Updates the the arhictecture of the last layer of the discriminator.

        Returns:
            nn.Sequential: The last layer of the discriminator.
        """
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dense_layer_params["dropout_p"]),
            nn.Linear(
                in_features=self.dense_layer_params["in_features"],
                out_features=self.dense_layer_params["out_features"],
            ),
        )

        self.criterion = nn.CrossEntropyLoss()

        return last_layer


class A2OClf(BaseClf):
    """
    Classifier for apple2orange dataset. 
    From BaseClf, it differs in the last layer and loss function.

    Args:
        params: Sequence[Any]: Model architecture parameters.
        use_checkpoint (bool, optional): Whether to use gradient checkpointing for
            memory efficiency during training. Defaults to False.
    """
    def __init__(self, params: Sequence[Any], use_checkpoint: bool = False):
        self.input_size = (256, 256)
        self.in_channels_num = 3

        super().__init__(
            self.input_size,
            self.in_channels_num,
            params,
            use_checkpoint=use_checkpoint,
        )

    def _create_last_layer(self) -> nn.Sequential:
        """Create the last layer of the classifier.

        Returns:
            nn.Sequential: The last layer of the classifier.
        """
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dense_layer_params["dropout_p"]),
            nn.Linear(
                in_features=self.dense_layer_params["in_features"],
                out_features=10,
            ),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid(),
        )

        self.criterion = nn.BCELoss()

        return last_layer


class SvhnClf(Discriminator):
    """
    Classifier for SVHN dataset, differs from BaseClf in the way layers are created.
    Also inherits from Discriminator.
    Args:
        params: Sequence[Any]: Model architecture parameters.
        use_checkpoint (bool, optional): Whether to use gradient checkpointing for
            memory efficiency during training. Defaults to False.
    """
    def __init__(self, params: Sequence[Any], use_checkpoint: bool = False):
        self.input_size = (32, 32)
        self.in_channels_num = 3
        self.dropout_p = params[-1]["dropout_p"]

        super().__init__(
            self.input_size,
            self.in_channels_num,
            params[:-1],
            negative_slope=params[-1]["leaky_relu_neg_slope"],
            use_checkpoint=use_checkpoint,
        )

        self.criterion = nn.CrossEntropyLoss()

    def _create_stand_layer(
        self,
        params: Tuple[Dict[str, Any], Dict[str, Any]],
        in_channels: int,
        **kwargs: Dict[str, Any],
    ) -> nn.Sequential:
        """
        A standard layer of the classifiers. Each layer consists of two convolutional blocks.
        One has the role to extract features, the other to downsample the feature maps.

        Args:
            params (Tuple[Dict[str, Any], Dict[str, Any]]): Parameters for the two convolutional blocks.
            in_channels (int): Number of input channels.

        Raises:
            ValueError: If input_size is not provided in kwargs.

        Returns:
            nn.Sequential: The created layer.
        """
        first_block_params, second_block_params = params

        conv1_params = first_block_params["conv_params"]
        conv2_params = second_block_params["conv_params"]
        pool2_params = second_block_params["pool_params"]

        input_size_raw = kwargs.get("input_size", None)
        if (
            input_size_raw is None
            or not isinstance(input_size_raw, tuple)
            or len(input_size_raw) != 2
            or not all(isinstance(i, int) for i in input_size_raw)
        ):
            raise ValueError(
                "input_size must be provided for Conv2dExtended layer to compute output size."
            )

        # -------------------------------------
        # First Block
        input_size: Tuple[int, int] = input_size_raw

        conv1 = Conv2dExtended(in_channels, input_size=input_size, **conv1_params)
        batch_norm1 = nn.BatchNorm2d(conv1.out_channels)
        relu1 = nn.LeakyReLU(negative_slope=self.leaky_relu_neg_slope)

        # -------------------------------------
        # Second Block
        conv1_output_size = conv1.compute_output_size(input_size)
        conv2 = Conv2dExtended(
            conv1.out_channels, input_size=conv1_output_size, **conv2_params
        )
        batch_norm2 = nn.BatchNorm2d(conv2.out_channels)
        relu2 = nn.LeakyReLU(negative_slope=self.leaky_relu_neg_slope)

        conv2_output_size = conv2.compute_output_size(conv1_output_size)
        pool2 = MaxPool2dExtended(input_size=conv2_output_size, **pool2_params)
        dropout2 = nn.Dropout2d(p=self.dropout_p)
        # -------------------------------------

        layer = nn.Sequential(
            conv1, batch_norm1, relu1, conv2, batch_norm2, relu2, pool2, dropout2
        )

        return layer

    def _create_last_layer(self) -> nn.Sequential:
        """
        Builds a last layer of the SVHN classifier. 
        Responsible for mapping the extracted features to class scores.

        Returns:
            nn.Sequential: The last layer of the classifier.
        """

        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.dense_layer_params["in_features"],
                out_features=self.dense_layer_params["out_features"],
            ),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(
                in_features=self.dense_layer_params["out_features"],
                out_features=self.dense_layer_params["num_classes"],
            ),
        )

        return last_layer

    def get_last_layer_out_channels(self) -> int:
        """Get the number of output channels from the last layer."""

        # Second convolution
        out_channels: int = self.layers[-1][3].out_channels  # type: ignore (check what is happening: is it nn.Module or nn.Sequential)
        return out_channels

    @staticmethod
    def _compute_layer_output_size(
        layer: nn.Sequential, input_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        conv1_output_size = layer[0].compute_output_size(input_size)  # type: ignore (why not ok?)
        conv2_output_size = layer[3].compute_output_size(conv1_output_size)  # type: ignore (why not ok?)
        pool_output_size = layer[6].compute_output_size(conv2_output_size)  # type: ignore (why not ok?)

        return pool_output_size


def select_classifier(domain_name: str, params: Sequence[Any], use_checkpoint: bool = False) -> Any:
    """
    Initialize a classifier for given domain with the architecture described in params.
    For the ResNet-based classifiers, params is ignored.
    
    Args:
        domain_name (str): Name of the dataset.
        params (Sequence[Any]): Architecture parameters.
        use_checkpoint (bool, optional): Whether to use gradient checkpointing for
            memory efficiency during training. Defaults to False.

    Raises:
        ValueError: If no classifier model is loaded.

    Returns:
        Any: Initialized classifier model.
    """
    clf = None

    match domain_name:
        case "MNIST":
            clf = BaseClf(
                input_size=(28, 28),
                in_channels=1,
                params=params,
                use_checkpoint=use_checkpoint,
            )

        case "USPS":
            clf = BaseClf(
                input_size=(16, 16),
                in_channels=1,
                params=params,
                use_checkpoint=use_checkpoint,
            )

        case "SVHN":
            clf = SvhnClf(params=params, use_checkpoint=use_checkpoint)

        case "A2O":
            clf = BaseClf(input_size=(256, 256), in_channels=3, params=params, use_checkpoint=use_checkpoint)

        case "VISDA_SOURCE":
            clf = ResNet18(in_channels_num=3, num_classes=12)

        case name if name.startswith("OFFICE_31"):
            clf = ResNet18(in_channels_num=3, num_classes=31)

        case name if name.startswith("HOME_OFFICE"):
            clf = ResNet18(in_channels_num=3, num_classes=65)

    if clf is None:
        raise ValueError("No classifier model loaded.")

    return clf.to(utils.DEVICE)


class ClfTrainer:
    """
    Trainer for classifier models.
    
    Args:
        clf (Any): Classifier model to be trained.
        optimizer (str): Optimizer type.
        epochs (int): Number of training epochs.
        patience (int): Patience for early stopping.
        lr (float): Learning rate.
        betas (Tuple[float, float]): Estimated moments for optimizers.
        weight_decay (float): Weight decay for optimizer.
        use_scheduler (bool): Whether to use learning rate scheduler.
        scheduler_factor (float): Factor for ReduceLROnPlateau scheduler.
        scheduler_patience (int): Patience for ReduceLROnPlateau scheduler.
        scheduler_min_lr (float): Minimum learning rate for scheduler.
        run_optuna (bool): Whether the trainer is used in an Optuna trial.
    """
    def __init__(
        self,
        clf: Any,
        optimizer: str = "Adam",
        epochs: int = 50,
        patience: int = 5,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        use_scheduler: bool = False,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 5,
        scheduler_min_lr: float = 1e-6,
        run_optuna: bool = False,
    ):
        self.clf: Any = clf
        self.lr: float = lr
        self.betas: Tuple[float, float] = betas
        self.weight_decay: float = weight_decay

        self.optimizer = utils.init_optimizer(
            optimizer,
            list(self.clf.parameters()),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        # Initialize learning rate scheduler
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if self.use_scheduler:
            print('Initializing scheduler for classifier optimizer.')
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Monitor accuracy (higher is better)
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr
            )

        self.epochs = epochs
        self.patience = patience

        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []

        self.best_acc: float = 0.0

        self.run_optuna = run_optuna

    def run_loop(self, loader: DataLoader[Any], train: bool = True) -> Tuple[float, float]:
        """
        Running loop for one epoch over the provided DataLoader.

        Args:
            loader (DataLoader[Any]): DataLoader for the dataset.
            train (bool, optional): Whether to run in training mode (backpropagation enabled), or in evaluation. 
                Defaults to True.

        Returns:
            Tuple[float, float]: Tuple containing the average loss and accuracy.
        """
        loss_total = 0
        acc_total = 0
        num_samples = 0
        for x, y in loader:
            # Move batch to the same device as the model
            x = x.to(utils.DEVICE, non_blocking=True)
            y = y.to(utils.DEVICE, non_blocking=True)

            if isinstance(self.clf, A2OClf):
                y = y.float().unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.clf.forward(x)

            loss = self.clf.criterion(outputs, y)
            if train:
                loss.backward()
                self.optimizer.step()

            loss_total += (
                loss.item()
            )  # reduction='sum' -> already returns sum of the losses

            if isinstance(self.clf, A2OClf):
                preds = (outputs >= 0.5).float()
                acc = (preds == y).sum()

            else:
                preds = outputs.argmax(dim=1)
                acc = (preds == y).sum()

            acc_total += acc.item()
            num_samples += y.size(0)

        loss_total /= len(loader)
        acc_total /= num_samples

        return loss_total, acc_total

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        *,
        trial: Optional[optuna.Trial] = None,
    ) -> Any:
        """
        Full training of the classifier. 

        Args:
            train_loader (DataLoader[Any]): DataLoader for the training dataset.
            val_loader (DataLoader[Any]): DataLoader for the validation dataset.
            trial (Optional[optuna.Trial], optional): Optuna trial object for hyperparameter tuning. Defaults to None.

        Raises:
            ValueError: If run_optuna is True and trial is not provided.
            optuna.exceptions.TrialPruned: If the trial should be pruned based on intermediate results.

        Returns:
            The trained classifier model.
        """
        if self.run_optuna and trial is None:
            raise ValueError("If run_optuna is True, trial must be provided.")

        best_clf_state_dict: Optional[dict] = None
        best_val_acc = -np.inf
        patience_cnt: int = 0

        for epoch in range(self.epochs):
            start_time = time.time()

            # Training Phase
            _ = self.clf.train()
            train_loss, train_acc = self.run_loop(train_loader)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            # Validation Phase
            _ = self.clf.eval()
            with torch.no_grad():
                val_loss, val_acc = self.run_loop(val_loader, train=False)

            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            # Update learning rate scheduler based on validation accuracy
            if self.use_scheduler:
                self.scheduler.step(val_acc)

            end_time = time.time()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_clf_state_dict = copy.deepcopy(self.clf.state_dict())
                patience_cnt = 0

            else:
                patience_cnt += 1

                if patience_cnt >= self.patience:
                    print(f"Patience {patience_cnt} reached its limit {self.patience}.")
                    break

            if self.run_optuna:
                trial.report(val_acc, epoch)  # type: ignore
                if trial.should_prune():  # type: ignore
                    raise optuna.exceptions.TrialPruned()

            else:
                print(f"Epoch {epoch}:")
                print(f"\tTrain loss: {train_loss:.6f}, Train acc: {train_acc:.6f}")
                print(f"\tVal loss: {val_loss:.6f}, Val acc: {val_acc:.6f}")
                print(f"\tTook: {(end_time - start_time) / 60:.2f} min")
                print(f"\tPatience: {patience_cnt}")

        if best_clf_state_dict is not None:
            _ = self.clf.load_state_dict(best_clf_state_dict)

        self.best_acc = best_val_acc

        return self.clf

    def get_trainer_info(self):
        """Log training hyperparameters and results."""
        info = {
            "optimizer": self.optimizer.__class__.__name__,
            "epochs": self.epochs,
            "patience": self.patience,
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "use_scheduler": self.use_scheduler,
            "best_acc": self.best_acc,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }
        return info
