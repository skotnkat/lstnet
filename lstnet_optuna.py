import os
import argparse
import json
import optuna
from sympy import hyper
import train

from models.lstnet import LSTNET
from data_preparation import AugmentOps
from LstnetTrainer import TrainParams

import hyperparam_modes


def objective(trial, cmd_args: argparse.Namespace) -> float:
    """Objective function for Optuna hyperparameter optimization.

    Args:
        cmd_args (argparse.Namespace): Command line arguments.
        trial: Optuna trial object.

    Returns:
        float: Best validation accuracy achieved during training.
    """

    # Suggest hyperparameters or take the values from cmd args

    # Architecture hyperparameters
    with open(f"{cmd_args.params_file}", "r", encoding="utf-8") as file:
        params = json.load(file)

    # Training loss weights
    weights = cmd_args.weights

    # ---------------------------------------------------------------
    # Optimizing weights

    weights = hyperparam_modes.suggest_weights(trial, weights_sum=sum(weights))

    # ---------------------------------------------------------------
    # Augmentation Ops
    augm_ops = AugmentOps(
        rotation=cmd_args.rotation,
        zoom=cmd_args.zoom,
        shift=cmd_args.shift,
    )

    max_epoch = cmd_args.optuna_max_resource
    patience = cmd_args.patience
    optim_name = cmd_args.optim_name
    lr = cmd_args.learning_rate
    betas = tuple(cmd_args.betas)
    weight_decay = cmd_args.weight_decay

    # Training parameters
    train_params = TrainParams(
        max_epoch_num=max_epoch,
        max_patience=patience,
        optim_name=optim_name,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )

    trained_model, logs = train.run(
        cmd_args.first_domain,
        cmd_args.second_domain,
        supervised=cmd_args.supervised,
        params=params,
        weights=weights,
        run_validation=True,
        manual_seed=cmd_args.manual_seed,
        val_data_size=cmd_args.val_size,
        batch_size=cmd_args.batch_size,
        num_workers=cmd_args.num_workers,
        augm_ops=augm_ops,
        train_params=train_params,
        optuna=True,
        optuna_trial=trial
    )

    trial.set_user_attr("train_logs", logs)

    model_path = f"{cmd_args.output_folder}/optuna_models/model_{trial.number}.pth"
    trial.set_user_attr("model_path", model_path)

    trained_model.save_model(model_path)

    return logs["trainer_info"]["best_loss"]


def run_optuna_lstnet(cmd_args) -> LSTNET:
    """Run Optuna hyperparameter optimization for LSTNET model.

    Args:
        cmd_args (argparse.Namespace): Command line arguments.

    """

    os.makedirs(f"{cmd_args.output_folder}/optuna_models", exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=cmd_args.optuna_sampler_start_trials,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=cmd_args.optuna_min_resource,
        max_resource=cmd_args.optuna_max_resource,
        reduction_factor=cmd_args.optuna_reduction_factor,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///optuna_{cmd_args.optuna_study_name}.db",
        study_name=cmd_args.optuna_study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, cmd_args),
        n_trials=cmd_args.optuna_trials,
    )

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best validation loss: {study.best_value}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"\t{key}: {value}")

    best_model_path = study.best_trial.user_attrs["model_path"]
    trained_model = LSTNET.load_lstnet_model(best_model_path)

    trained_model.save_model(f"{cmd_args.output_folder}/{cmd_args.model_file_name}")

    all_logs = {}
    for i, trial in enumerate(study.trials):
        all_logs[f"trial_{i}"] = trial.user_attrs["train_logs"]

    all_logs["optuna_study_info"] = {
        "best_trial_number": study.best_trial.number,
        "best_trial_value": study.best_trial.value,
        "total_trials_completed": len(study.trials),
        "database_file": f"optuna_{cmd_args.optuna_study_name}.db",
        "study_name": cmd_args.optuna_study_name,
    }

    with open(f"{cmd_args.output_folder}/{cmd_args.logs_file_name}", "w") as file:
        json.dump(all_logs, file, indent=2)

    return trained_model
