"""Utilities for training, evaluating and hyperparamter tuning of classifiers."""

from typing import Any, Tuple, List, Optional
import json
from torch.utils.data import DataLoader
import optuna

from eval_models.clf_models import select_classifier, ClfTrainer, BaseClf
import utils

from data_preparation import load_augmented_dataset, AugmentOps, ResizeOps


def prepare_clf_data(
    domain_name: str,
    *,
    val_size_data: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    augment_ops: Optional[AugmentOps] = None,
    resize_ops: Optional[ResizeOps] = None,
    inplace_augmentation: bool = False,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Prepare data loaders for classifier training and validation.

    Args:
        domain_name (str): Name of the dataset.
        val_size_data (float): Proportion of data to use for validation.
        seed (int): Random seed for data splitting.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        augment_ops (Optional[AugmentOps], optional): Augmentation operations to apply. 
            Defaults to None.
        resize_ops (Optional[ResizeOps], optional): Resize operations to apply. 
            Defaults to None.
        inplace_augmentation (bool, optional): Whether to apply augmentation in-place. 
            Defaults to False.

    Returns:
        Tuple[DataLoader[Any], DataLoader[Any]]: Training and validation data loaders.
    """
    
    ds_name: str = domain_name.upper()
    val_size: float = val_size_data
    manual_seed: int = seed
    train_data, val_data = load_augmented_dataset(
        ds_name,
        split_data=True,
        train_op=True,
        val_data_size=val_size,
        manual_seed=manual_seed,
        augment_ops=augment_ops,
        resize_ops=resize_ops,
        inplace_augmentation=inplace_augmentation,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(
            True if utils.DEVICE is not None and utils.DEVICE.type == "cuda" else False
        ),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            True if utils.DEVICE is not None and utils.DEVICE.type == "cuda" else False
        ),
    )

    return train_loader, val_loader


def get_clf(
    domain_name: str,
    *,
    params_path: Optional[str] = None,
    clf_params: Optional[List[Any]] = [],
):
    """
    Get classifier model based on domain name and parameters.

    Args:
        domain_name (str): Domain name to select the type of classifier.
        params_path (Optional[str], optional): Path to the parameters file. 
            Defaults to None.
        clf_params (Optional[List[Any]], optional): Classifier parameters. Used for hyperparameter tuning.
            Defaults to [].

    Returns:
        Classifier model instance.
    """

    if clf_params is not None:
        params = clf_params
    else:
        with open(f"{params_path}", "r", encoding="utf-8") as file:
            params = json.load(file)
    clf = select_classifier(domain_name.upper(), params=params)

    return clf


def train_clf(
    clf: Any,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    *,
    optim: str,
    epoch_num: int,
    patience: int,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    use_scheduler: bool = False,
    scheduler_factor: float = 0.1,
    scheduler_patience: int = 5,
    scheduler_min_lr: float = 1e-6,
    optuna_trial: Optional[optuna.Trial] = None
):
    """
    Train classifier on the training loader. Select the best model based on validation-accuracy.

    Args:
        clf (Any): Classifier model instance.
        train_loader (DataLoader[Any]): Training data loader.
        val_loader (DataLoader[Any]): Validation data loader.
        optim (str): Optimizer type.
        epoch_num (int): Number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping (on validation accuracy).
        lr (float): Learning rate.
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
        weight_decay (float): Weight decay (L2 penalty).
        use_scheduler (bool, optional): Whether to use a learning rate scheduler. Defaults to False.
        scheduler_factor (float, optional): Factor by which the learning rate will be reduced. Defaults to 0.1.
        scheduler_patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Defaults to 5.
        scheduler_min_lr (float, optional): A lower bound on the learning rate of all param groups. Defaults to 1e-6.
        optuna_trial (Optional[optuna.Trial], optional): Optuna trial object for hyperparameter tuning. Defaults to None.

    Returns:
        Tuple[Any, float, dict]: Trained classifier, best validation accuracy, and training information.
    """
    trainer = ClfTrainer(
        clf,
        optimizer=optim,
        epochs=epoch_num,
        patience=patience,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        use_scheduler=use_scheduler,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
        run_optuna=optuna_trial is not None,
    )

    clf = trainer.train(train_loader, val_loader, trial=optuna_trial)
    print(f"Best validation accuracy: {trainer.best_acc}")
    
    return (
        clf,
        trainer.best_acc,
        trainer.get_trainer_info(),
    )
