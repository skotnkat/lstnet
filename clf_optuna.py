import optuna
import time


import clf_utils



def suggest_architecture(trial):
    num_stages = trial.suggest_categorical("num_stages", [3, 4])
    base = trial.suggest_categorical("base_channels", [16, 32, 64])
    conv_first_kernel = trial.suggest_categorical("conv_first_kernel", [3, 5])
    stage_channels = [base * (2**i) for i in range(num_stages)]

    pool_down_kernel = trial.suggest_categorical("pool_down_kernel", [2, 3])

    params = []

    for out_channels in stage_channels:
        conv1 = {
            "out_channels": out_channels,
            "kernel_size": conv_first_kernel,
            "stride": 1,
            "padding": "same",
        }

        pool1 = {"kernel_size": 3, "stride": 1, "padding": "same"}

        conv2 = {
            "out_channels": out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        pool2 = {
            "kernel_size": pool_down_kernel,
            "stride": 2,
            "padding": "same",
        }

        params.append([conv1, pool1])
        params.append([conv2, pool2])

    dropout_p = trial.suggest_float("dropout_prob", 0.1, 0.5)
    params.append(
        {"out_features": 10, "dropout_p": dropout_p}
    )  # !!! Fix hardcoded out channels (can be deduced from dataset)

    leaky_relu_neg_slope = trial.suggest_float("leaky_relu_neg_slope", 0.01, 0.3)
    params.append({"leaky_relu_neg_slope": leaky_relu_neg_slope})

    return params


def objective(trial, cmd_args):
    start_time = time.time() 
    params = suggest_architecture(trial)

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

    clf = clf_utils.get_clf(cmd_args.domain_name, clf_params=params)

    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    beta1 = trial.suggest_categorical("beta1", [0.85, 0.9, 0.95])
    beta2 = trial.suggest_categorical("beta2", [0.99, 0.999, 0.9999])

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
        optuna_trial=trial,
    )

    end_time = time.time()
    print(f"Trial took: {(end_time - start_time) / 60:.2f} s")
    
    trial.set_user_attr("architecture_params", params)
     
    return best_acc


def rerun_with_best_params(best_params, best_architecture_params, cmd_args):
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

    clf = clf_utils.get_clf(cmd_args.domain_name, clf_params=best_architecture_params)

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

    # Add Optuna information to trainer_info
    trainer_info["optuna_best_params"] = best_params
    trainer_info["optuna_config"] = {
        "study_name": cmd_args.study_name,
        "n_trials": cmd_args.n_trials,
        "optuna_sampler_start_trials": cmd_args.optuna_sampler_start_trials,
        "min_resource": cmd_args.min_resource,
        "max_resource": cmd_args.max_resource,
        "reduction_factor": cmd_args.reduction_factor,
        "final_best_accuracy": best_acc,
    }

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

    trained_clf, trainer_info = rerun_with_best_params(
        study.best_trial.params,
        study.best_trial.user_attrs["architecture_params"],
        cmd_args,
    )

    # Add study information to trainer_info
    trainer_info["optuna_study_info"] = {
        "best_trial_number": study.best_trial.number,
        "best_trial_value": study.best_trial.value,
        "total_trials_completed": len(study.trials),
        "database_file": f"optuna_clf_{cmd_args.study_name}.db",
    }

    return trained_clf, trainer_info
