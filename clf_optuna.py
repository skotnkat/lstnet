import optuna
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_parallel_coordinate, plot_contour
import matplotlib.pyplot as plt
import random
import os
import argparse
import utils
import json
import train_eval_clf
import clf_utils
import torch
import shutil


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('domain_name', type=str.upper)
    parser.add_argument('--custom_clf', action="store_true")
    parser.add_argument("--input_size", type=int, nargs=2, default=(28, 28))
    parser.add_argument("--in_channels", type=int, default=1)

    return parser.parse_args()


def generate_conv_pool_params(trial, num_layers):
    stride2count = trial.suggest_int("stride2_count", 0, min(2, num_layers))
    stride2_layers = random.sample(range(num_layers), stride2count)

    layers = []
    for i in range(num_layers):
        conv_params = {
            "out_channels": trial.suggest_categorical(f"neurons_layer_{i}", [64, 128, 256, 512]),
            "kernel_size": 3,
            "stride": 1,
            "padding": "same"
        }

        is_stride2_layer = (i in stride2_layers)
        padding_value = 0 if is_stride2_layer else "same"
        pool_params = {
            "kernel_size": 2,
            "stride": is_stride2_layer + 1,
            "padding": padding_value
        }

        layers.append([conv_params, pool_params])

    return layers


def objective(trial, domain_name, input_size, in_channels, custom_clf):
    rotation = trial.suggest_int("augm_rotation", 10, 30, step=5)
    zoom = trial.suggest_float("augm_zoom", 0.1, 0.3, step=0.05)
    shift = trial.suggest_int("augm_shift", 2, 5)  # in pixels

    train_loader, val_loader = train_eval_clf.load_data(args.domain_name, rotation=rotation, zoom=zoom, shift=shift)

    # --- Sample hyperparameters ---
    num_layers = trial.suggest_int("num_layers", 4, 8)
    conv_layers_params = generate_conv_pool_params(trial, num_layers)

    # dense layer
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    leaky_relu = trial.suggest_float("leaky_relu", 0.01, 0.3)
    dense_layer_params = {
        "out_features": 10,  # MNIST output
        "dropout_p": dropout
    }

    model_params = conv_layers_params + [dense_layer_params]

    clf = clf_utils.select_classifier(domain_name, input_size, in_channels, model_params, leaky_relu, custom_clf)

    # optimizer + training
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"])
    patience = trial.suggest_int("patience", 5, 20, step=5)
    epoch_num = trial.suggest_categorical("epoch_num", [25, 50, 100])

    clf.optimizer = utils.init_optimizer(optimizer_name, clf.parameters(), lr)
    clf.epochs = epoch_num
    clf.patience = patience

    trained_clf, val_acc, results = train_eval_clf.train(clf, train_loader, val_loader, run_optuna=True, trial=trial)

    model_path = f"{output_dir}/clf_model_trial_{trial.number}.pth"
    torch.save(trained_clf, model_path)

    trial.set_user_attr("model_path", model_path)

    return val_acc


if __name__ == "__main__":
    utils.assign_device()
    args = parse_args()
    output_dir = f"optuna_results/{args.domain_name}"
    os.makedirs(output_dir, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=30, interval_steps=1)

    sampler = optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                storage=f"sqlite:///optuna_clf_{args.domain_name}.db",
                                study_name=args.domain_name, load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args.domain_name, args.input_size, args.in_channels, args.custom_clf),
                   n_trials=100)

    best_trial = study.best_trial
    best_val_acc = best_trial.value
    best_params = best_trial.params
    best_model_path = best_trial.user_attrs["model_path"]

    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best params:')
    for key, value in best_params.items():
        print(f"\t{key}: {value}")

    best_params["best_val_acc"] = best_val_acc
    with open(f"{output_dir}/params.json", "w") as file:
        json.dump(best_params, file, indent=2)

    shutil.copy(best_model_path, f"{output_dir}/best_model.pth")
    # Cleanup: remove all other .pth files except best_model.pth
    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)
        if filename.endswith(".pth") and os.path.abspath(path) not in [
            os.path.abspath(best_model_path),
            os.path.abspath(f"{output_dir}/best_model.pth")
        ]:
            try:
                os.remove(path)
                print(f"Deleted: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")