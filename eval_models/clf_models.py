from typing import Tuple, Sequence, Any, Optional, List
import time
import numpy as np
import copy

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

import optuna

from models.discriminator import Discriminator
import utils


class BaseClf(Discriminator):
    def __init__(
        self,
        input_size: Tuple[int, int],
        in_channels: int,
        params: Sequence[Any],
    ):
        self.input_size = input_size
        self.in_channels_num = in_channels

        super().__init__(
            self.input_size,
            self.in_channels_num,
            params[:-1],
            negative_slope=params[-1]["leaky_relu_neg_slope"],
        )

        self.criterion = nn.CrossEntropyLoss()

    def _create_last_layer(self):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dense_layer_params["dropout_p"]),
            nn.Linear(
                in_features=self.dense_layer_params["in_features"],
                out_features=self.dense_layer_params["out_features"],
            ),
        )

        return last_layer


# class SvhnClf(Discriminator):
#     def __init__(self, params):
#         self.input_size = (32, 32)
#         self.in_channels_num = 3
#         self.global_params = params[-1]
#         super().__init__(self.input_size, self.in_channels_num, params[:-1])
#
#         self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
#         self.criterion = nn.CrossEntropyLoss()
#         self.epochs = 50
#         self.patience = 10
#
#
#     def _create_stand_layer(self, params, in_channels, input_size):
#         conv_params, pool_params = params
#         conv = Conv2dExtended(in_channels, input_size=input_size, **conv_params)
#         batch_norm = nn.BatchNorm2d(conv_params["out_channels"], momentum=self.global_params["batch_norm_momentum"])
#         relu = nn.ReLU()
#
#         layers = [conv, batch_norm, relu]
#
#         if len(pool_params):
#             output_size = conv.compute_output_size(input_size)
#             layers.append(MaxPool2dExtended(input_size=output_size, **pool_params))
#             layers.append(negative_slope=self.global_params["leaky_relu_neg_slope"])
#
#         return nn.Sequential(*layers)
#
#     def _create_last_layer(self):
#         last_layer = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(self.dense_layer_params["dropout_p"]),
#             nn.Linear(in_features=self.dense_layer_params["in_features"],
#                       out_features=self.dense_layer_params["out_features"])
#         )
#
#         return last_layer


def select_classifier(domain_name, params):
    clf = None

    match domain_name:
        case "MNIST":
            clf = BaseClf(
                input_size=(28, 28),
                in_channels=1,
                params=params,
            )

        case "USPS":
            clf = BaseClf(
                input_size=(16, 16),
                in_channels=1,
                params=params,
            )

    if clf is None:
        raise ValueError("No classifier model as loaded.")

    return clf.to(utils.DEVICE)


class ClfTrainer:
    def __init__(
        self,
        clf: BaseClf,
        optimizer: str = "Adam",
        epochs: int = 50,
        patience: int = 5,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        run_optuna: bool = False,
    ):
        self.clf: BaseClf = clf
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

        self.epochs = epochs
        self.patience = patience

        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []

        self.best_acc: float = 0.0

        self.run_optuna = run_optuna

    def run_loop(self, loader: DataLoader[Any], train: bool = True):
        loss_total = 0
        acc_total = 0
        num_samples = 0
        for x, y in loader:
            # Move batch to the same device as the model
            x = x.to(utils.DEVICE, non_blocking=True)
            y = y.to(utils.DEVICE, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.clf.forward(x)

            loss = self.clf.criterion(outputs, y)
            if train:
                loss.backward()
                self.optimizer.step()

            loss_total += (
                loss.item()
            )  # reduction='sum' -> already returns sum of the losses

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
    ):
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
        info = {
            "optimizer": self.optimizer.__class__.__name__,
            "epochs": self.epochs,
            "patience": self.patience,
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "best_acc": self.best_acc,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }
        return info
