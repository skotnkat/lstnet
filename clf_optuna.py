import optuna
import torch

import clf_utils


def objective(trial, cmd_args):
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    beta1 = trial.suggest_categorical("beta1", [0.85, 0.9, 0.95])
    beta2 = trial.suggest_categorical("beta2", [0.99, 0.999, 0.9999])

    rot = trial.suggest_int("augm_rotation", 5, 20, step=5)
    zoom = trial.suggest_float("augm_zoom", 0.05, 0.2, step=0.05)
    shift = trial.suggest_int("augm_shift", 1, 4)  # in pixels

    train_loader, val_loader = clf_utils.prepare_clf_data(
        cmd_args.domain_name,
        val_size_data=cmd_args.val_size,
        seed=cmd_args.manual_seed,
        batch_size=cmd_args.batch_size,
        num_workers=cmd_args.num_workers,
        rotation=rot,
        zoom=zoom,
        shift=shift,
    )

    clf = clf_utils.get_clf(cmd_args.domain_name, cmd_args.params_file)

    _, best_acc, _ = clf_utils.train_clf(
        clf,
        train_loader,
        val_loader,
        optim=cmd_args.optimizer,
        epoch_num=cmd_args.max_resource,
        patience=cmd_args.epoch_num,
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=wd,
    )

    return best_acc


def rerun_with_best_params(best_params, cmd_args):
    train_loader, val_loader = clf_utils.prepare_clf_data(
        cmd_args.domain_name,
        val_size_data=cmd_args.val_size,
        seed=cmd_args.manual_seed,
        batch_size=cmd_args.batch_size,
        num_workers=cmd_args.num_workers,
        rotation=best_params["augm_rotation"],
        zoom=best_params["augm_zoom"],
        shift=best_params["augm_shift"],
    )

    clf = clf_utils.get_clf(cmd_args.domain_name, cmd_args.params_file)

    trained_clf, best_acc, trainer_info = clf_utils.train_clf(
        clf,
        train_loader,
        val_loader,
        optim=cmd_args.optimizer,
        epoch_num=cmd_args.max_resource * 2,  # Train longer
        patience=cmd_args.epoch_num,
        lr=best_params["lr"],
        betas=(best_params["beta1"], best_params["beta2"]),
        weight_decay=best_params["weight_decay"],
    )

    print(f"Final best accuracy after retraining: {best_acc:.4f}")

    return trained_clf, trainer_info


def run_optuna_clf(cmd_args):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=cmd_args.optuna_sampler_start_trials,
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=cmd_args.min_resource,
        max_resource=cmd_args.max_resource,
        reduction_factor=cmd_args.reduction_factor,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///optuna_clf_{cmd_args.study_name}.db",
        study_name=cmd_args.study_name,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, cmd_args),
        n_trials=cmd_args.n_trials,
    )

    print(f"Best validation accuracy: {study.best_trial.value:.4f}")
    print(f"Best params:")
    for key, value in study.best_trial.params.items():
        print(f"\t{key}: {value}")

    return rerun_with_best_params(study.best_trial.params, cmd_args)
