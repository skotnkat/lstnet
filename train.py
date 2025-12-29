"""
Module is implementing the training procedure of LSTNET model from start to finish.
It includes data loading, model initialization, training, validation,
and saving the trained model along with training logs.
"""

from typing import List, Dict, Any, Optional, cast, Union, Tuple, overload, Literal
import json

import os
import torch
from torch.utils.data import DataLoader
import optuna

from dual_domain_dataset import DualDomainDataset
from models.lstnet import LSTNET
from data_preparation import get_training_loader, AugmentOps, ResizeOps
from LstnetTrainer import LstnetTrainer, TrainParams
import utils


@overload
def run(
    first_domain_name: str,
    second_domain_name: str,
    *,
    supervised: bool,
    params: Dict[str, Any],
    weights: List[float],
    run_validation: bool = True,
    output_folder: str = "results/",
    model_file_name: str = "lstnet.pth",
    logs_file_name: str = "loss_logs.json",
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    augm_ops: Optional[AugmentOps] = None,
    resize_ops: Optional[ResizeOps] = None,
    train_params: TrainParams = TrainParams(),
    optuna: Literal[False] = False,
    optuna_trial: Optional[optuna.Trial] = None,
    compile_model: bool = False,
    use_checkpoint: bool = False,
) -> LSTNET: ...
@overload
def run(
    first_domain_name: str,
    second_domain_name: str,
    *,
    supervised: bool,
    params: Dict[str, Any],
    weights: List[float],
    run_validation: bool = True,
    output_folder: str = "results/",
    model_file_name: str = "lstnet.pth",
    logs_file_name: str = "loss_logs.json",
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    augm_ops: Optional[AugmentOps] = None,
    resize_ops: Optional[ResizeOps] = None,
    train_params: TrainParams = TrainParams(),
    optuna: Literal[True],
    optuna_trial: Optional[optuna.Trial] = None,
    compile_model: bool = False,
    use_checkpoint: bool = False,
) -> Tuple[LSTNET, Dict[str, Any]]: ...
def run(
    first_domain_name: str,
    second_domain_name: str,
    *,
    supervised: bool,
    params: Dict[str, Any],
    weights: List[float],
    run_validation: bool = True,
    output_folder: str = "results/",
    model_file_name: str = "lstnet.pth",
    logs_file_name: str = "loss_logs.json",
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    augm_ops: Optional[AugmentOps] = None,
    resize_ops: Optional[ResizeOps] = None,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    train_params: TrainParams = TrainParams(),
    optuna: bool = False,
    optuna_trial: Optional[optuna.Trial] = None,
    compile_model: bool = False,
    use_checkpoint: bool = False,
    use_svhn_extra: bool = False,
) -> Union[LSTNET, Tuple[LSTNET, Dict[str, Any]]]:
    """Train the LSTNET model.

    Args:
        first_domain_name (str): Name of the first domain.
        second_domain_name (str): Name of the second domain.
        supervised (bool): Whether to use supervised training.
        params (Dict[str, Any]): Model parameters.
        weights (List[float]): Loss weights.
        run_validation (bool, optional): Whether to run validation. Defaults to True.
        output_folder (str, optional): Folder to save the results. Defaults to "results/".
        model_file_name (str, optional): File name for the saved model. Defaults to "lstnet.pth".
        logs_file_name (str, optional): _description_. Defaults to "loss_logs.json".
        manual_seed (int, optional): _description_. Defaults to 0.4.
        val_data_size (float, optional): _description_. Defaults to 0.4.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        augm_ops (AugmentOps, optional): Data augmentation operations. Defaults to AugmentOps().
        train_params (TrainParams, optional): Training hyperparameters. Defaults to TrainParams().
        use_checkpoint (bool, optional): Enable gradient checkpointing. Defaults to False.

    Returns:
        LSTNET: The trained LSTNET model.
    """

    train_loader: DataLoader[DualDomainDataset]
    val_loader: Optional[DataLoader[DualDomainDataset]] = None

    pin_memory = True if torch.cuda.is_available() else False

    if run_validation:
        train_loader, val_loader = get_training_loader(
            first_domain_name,
            second_domain_name,
            supervised,
            split_data=True,
            manual_seed=manual_seed,
            val_data_size=val_data_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augment_ops=augm_ops,
            skip_augmentation=skip_augmentation,
            resize_ops=resize_ops,
            use_svhn_extra=use_svhn_extra,
        )
    else:
        train_loader = get_training_loader(
            first_domain_name,
            second_domain_name,
            supervised,
            split_data=False,
            manual_seed=manual_seed,
            val_data_size=val_data_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augment_ops=augm_ops,
            skip_augmentation=skip_augmentation,
            resize_ops=resize_ops,
            use_svhn_extra=use_svhn_extra,
        )

        val_loader = None

    # Cast access to DualDomainDataset specific methods
    dds = cast(DualDomainDataset, train_loader.dataset)
    (first_channels, first_h, first_w), (second_channels, second_h, second_w) = (
        dds.get_input_dims()
    )

    model = LSTNET(
        first_domain_name,
        second_domain_name,
        params,
        first_input_size=(first_h, first_w),
        second_input_size=(second_h, second_w),
        first_in_channels_num=first_channels,
        second_in_channels_num=second_channels,
        use_checkpoint=use_checkpoint,
    )

    utils.init_logs(["train", "val"])
    trainer = LstnetTrainer(
        model,
        weights,
        train_loader,
        val_loader=val_loader,
        train_params=train_params,
        run_optuna=optuna,
        optuna_trial=optuna_trial,
        compile_model=compile_model,
    )
    print("Starting train and validate")
    trained_model = trainer.fit()

    trainer_info = trainer.get_trainer_info()
    trainer_info["first_domain"] = first_domain_name
    trainer_info["second_domain"] = second_domain_name
    trainer_info["supervised"] = supervised
    trainer_info["augmentation_ops"] = str(augm_ops)
    trainer_info["batch_size"] = batch_size
    trainer_info["train_params"] = str(train_params)

    utils.LOSS_LOGS["trainer_info"] = trainer_info

    if optuna:
        return trained_model, utils.LOSS_LOGS.copy()

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    model_path = f"{output_folder}/{model_file_name}"
    trained_model.save_model(model_path)

    with open(f"{output_folder}/{logs_file_name}", "w", encoding="utf-8") as f:
        json.dump(utils.LOSS_LOGS, f, indent=2)

    return trained_model
