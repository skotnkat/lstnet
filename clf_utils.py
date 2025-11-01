from typing import Any, Tuple, List, Optional
import json
from torch.utils.data import DataLoader
import optuna

from eval_models.clf_models import select_classifier, ClfTrainer, BaseClf
import utils

from data_preparation import load_augmented_dataset, AugmentOps


def prepare_clf_data(
    domain_name: str,
    *,
    val_size_data: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    rotation: int,
    zoom: float,
    shift: int,
):
    # Load Trainining and Validation Data
    aug_ops = AugmentOps(rotation=rotation, zoom=zoom, shift=shift)

    ds_name: str = domain_name.upper()
    val_size: float = val_size_data
    manual_seed: int = seed
    train_data, val_data = load_augmented_dataset(
        ds_name,
        split_data=True,
        train_op=True,
        val_data_size=val_size,
        manual_seed=manual_seed,
        augment_ops=aug_ops,
    )
    print("Dataset Loaded.")

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
    print("DataLoaders Created.")

    return train_loader, val_loader


def get_clf(
    domain_name: str,
    *,
    params_path: Optional[str] = None,
    clf_params: Optional[List[Any]] = None,
):
    # Load Parameters File
    if params_path is None and clf_params is None:
        raise ValueError("Either params_path or clf_params must be provided.")

    if clf_params is not None:
        params = clf_params
    else:
        with open(f"{params_path}", "r", encoding="utf-8") as file:
            params = json.load(file)

    clf = select_classifier(domain_name.upper(), params=params)
    print("Classifier Selected.")

    return clf


def train_clf(
    clf: BaseClf,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    *,
    optim: str,
    epoch_num: int,
    patience: int,
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float,
    optuna_trial: Optional[optuna.Trial] = None
):
    trainer = ClfTrainer(
        clf,
        optimizer=optim,
        epochs=epoch_num,
        patience=patience,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        run_optuna=optuna_trial is not None,
    )

    clf = trainer.train(train_loader, val_loader, trial=optuna_trial)
    
    print(f"Best validation accuracy: {trainer.best_acc}")
    
    return (
        clf,
        trainer.best_acc,
        trainer.get_trainer_info(),
    )
