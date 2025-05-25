import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import random
import os
import argparse
import utils
import train_eval_clf
import clf_utils
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('domain_name', type=str.upper)

    return parser.parse_args()


def generate_conv_pool_params(trial, num_layers):
    stride2count = trial.suggest_int("stride2_count", 0, min(2, num_layers))
    stride2_layers = random.sample(range(num_layers), stride2count)

    layers = []
    for i in range(num_layers):
        conv_params = {
            "out_channels": trial.suggest_int(f"neurons_layer_{i}", 64, 256, step=64),
            "kernel_size": 3,
            "stride": 1,
            "padding": "same"
        }

        is_stride2_layer = (i in stride2_layers)
        pool_params = {
            "kernel_size": 2,
            "stride": is_stride2_layer + 1,
            "padding": 0 if is_stride2_layer else "same"
        }

        layers.append([conv_params, pool_params])

    return layers


def objective(trial, domain_name):
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

    model_params = conv_layers_params + [dense_layer_params] + [{"leaky_relu_neg_slope": leaky_relu}]

    # if other -> change
    utils.assign_device()
    model = clf_utils.select_classifier(domain_name, model_params)

    # optimizer + training
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"])
    epochs = trial.suggest_int("epochs", 10, 100, step=10)
    patience = trial.suggest_int("patience", 5, 20, step=5)

    model.optimizer = utils.init_optimizer(optimizer_name, model.parameters(), lr)
    model.epochs = epochs
    model.patience = patience

    train_loader, val_loader = train_eval_clf.load_data(domain_name)

    clf, val_acc, results = train_eval_clf.train(domain_name, model_params, train_loader, val_loader, optuna=True, trial=trial)

    trial.set_user_attr("best_model", clf)
    return val_acc


if __name__ == "__main__":
    utils.assign_device()
    args = parse_args()

    sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, group=True)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args.domain_name), n_trials=100)

    trial = study.best_trial
    print(f'Best validation accuracy: {trial.value:.4f}')
    print(f'Best params:')
    best_params = trial.params
    for key, value in best_params.items():
        print(f"\t{key}: {value}")

    output_dir = f"optuna_results/{args.domain_name}"
    os.makedirs(output_dir, exist_ok=True)
    # visualize importances and accuracies
    importances_fig = plot_param_importances(study)
    importances_fig.savefig(f"{output_dir}/param_importances.png")

    acc_fig = plot_optimization_history(study)
    acc_fig.savefig(f"{output_dir}/optimization_history.png")

    best_model = study.best_trial.user_attrs["best_model"]
    torch.save(best_model, f"{output_dir}/model.pth")
