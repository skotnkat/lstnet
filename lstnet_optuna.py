import copy

import optuna
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_parallel_coordinate
import matplotlib.pyplot as plt
import os
import argparse

import data_preparation
import utils
import json
import train
from models.lstnet import LSTNET
import torch
import loss_functions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('first_domain', type=str.upper)
    parser.add_argument('second_domain', type=str.upper)
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")
    parser.add_argument("--supervised", action="store_true")

    return parser.parse_args()


def get_stand_conv_params(out_channels, kernel_size, stride=1, padding="same"):
    return {
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding
    }


def get_stand_max_pool_params(kernel_size, stride=1, padding="same"):
    return {
        "kernel_size":  kernel_size,
        "stride":  stride,
        "padding":  padding}


def update_disc_params(trial, orig_layer_params):
    new_layer_params = copy.deepcopy(orig_layer_params)

    base = trial.suggest_categorical("stand_disc_base_base_out_channels", [32, 64, 128])
    extra_layer = trial.suggest_categorical("stand_disc_extra_layer", [True, False])

    stand_layers_num = len(new_layer_params["first_discriminator"])

    for i in range(stand_layers_num-1):
        new_layer_params["first_discriminator"][i][0]["out_channels"] = base
        new_layer_params["second_discriminator"][i][0]["out_channels"] = base

        base *= 2

    if extra_layer:
        kernel_size = trial.suggest_categorical("stand_disc_extra_layer_kernel_size", [2, 3, 5])
        extra_conv = get_stand_conv_params(base, kernel_size)
        max_pool_params = get_stand_max_pool_params(kernel_size)

        new_layer_params["first_discriminator"].insert(-1, [extra_conv, max_pool_params])
        new_layer_params["second_discriminator"].insert(-1, [extra_conv, max_pool_params])

    shared_layers_num = trial.suggest_int("enc_gen_shared_layers_num", 3, 5)
    kernel_size = trial.suggest_categorical("enc_gen_shared_kernel_size", [3, 5])

    latent_disc_params = []
    for i in range(shared_layers_num):
        out_channels = trial.suggest_categorical(f"shared_disc_out_channels_{i}", [128, 256, 512])
        conv_params = get_stand_conv_params(out_channels, kernel_size)
        max_pool_params = get_stand_max_pool_params(kernel_size)

        latent_disc_params.append([conv_params, max_pool_params])

    # with out channels
    latent_disc_params.append(orig_layer_params["latent_discriminator"][-1])

    new_layer_params["latent_discriminator"] = latent_disc_params

    return new_layer_params


# update to be able to introduce asymetry (new function)
def update_enc_gen_params(trial, orig_layer_params):
    base = trial.suggest_categorical("stand_enc_gen_base_base_out_channels", [32, 64, 128])

    extra_layer = trial.suggest_categorical("stand_enc_gen_extra_layer", [True, False])

    new_layer_params = copy.deepcopy(orig_layer_params)
    stand_layers_num = len(new_layer_params["first_encoder"])

    for enc_idx in range(stand_layers_num):
        new_layer_params["first_encoder"][enc_idx]["out_channels"] = base
        new_layer_params["second_encoder"][enc_idx]["out_channels"] = base

        gen_idx = -enc_idx - 1
        new_layer_params["first_generator"][gen_idx]["out_channels"] = base
        new_layer_params["second_generator"][gen_idx]["out_channels"] = base

        base *= 2

    if extra_layer:
        kernel_size = trial.suggest_categorical("stand_enc_gen_extra_layer_kernel_size", [3, 5, 7])
        extra_conv = get_stand_conv_params(base, kernel_size)

        new_layer_params["first_encoder"].append(extra_conv)
        new_layer_params["second_encoder"].append(extra_conv)

        new_layer_params["first_generator"].insert(0, extra_conv)
        new_layer_params["second_generator"].insert(0, extra_conv)


    # shared encoder generator
    shared_layers_num = trial.suggest_int("enc_gen_shared_layers_num", 2, 4)
    kernel_size = trial.suggest_categorical("enc_gen_shared_kernel_size", [3, 5])

    last_out_channels = new_layer_params["first_encoder"][-1]["out_channels"]
    out_channels = trial.suggest_categorical("enc_gen_shared_base_out_channels", [last_out_channels, last_out_channels/2, last_out_channels/4])

    shared_encoder = []
    shared_generator = []
    for enc_idx in range(shared_layers_num):
        layer = get_stand_conv_params(base, kernel_size)
        shared_encoder.append(layer)
        shared_generator.insert(0, layer)

        base = base // 2  # / -> result is float

    new_layer_params["shared_encoder"] = shared_encoder
    new_layer_params["shared_generator"] = shared_generator
    
    return new_layer_params


def objective(trial, first_domain, second_domain, orig_layer_params, val_loader, train_loader):
    updated_layer_params = update_enc_gen_params(trial, orig_layer_params)
    fin_layer_params = update_disc_params(trial, updated_layer_params)

    leaky_relu_neg_slope = trial.suggest_float("negative_slope", 0.01, 0.3)
    batch_norm_momentum = trial.suggest_float("momentum", 0.01, 0.3, step=0.01)
    fin_layer_params["leaky_relu"] = {"negative_slope": leaky_relu_neg_slope}
    fin_layer_params["batch_norm"] = {"momentum": batch_norm_momentum}

    loss_functions.W1 = trial.suggest_int("w1", 20, 100, step=10)
    loss_functions.W2 = trial.suggest_int("w2", 20, 100, step=10)
    loss_functions.W3 = trial.suggest_int("w3", 20, 100, step=10)
    loss_functions.W4 = trial.suggest_int("w4", 20, 100, step=10)
    loss_functions.W5 = trial.suggest_int("w5", 20, 100, step=10)
    loss_functions.W6 = trial.suggest_int("w6", 20, 100, step=10)

    epochs = trial.suggest_int("epochs", 50, 150, step=50)
    patience = trial.suggest_int("patience", 5, 20, step=5)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"])

    beta1 = trial.suggest_float("beta1", 0.85, 0.95)
    beta2 = trial.suggest_float("beta2", 0.98, 0.999)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.01)

    utils.OPTIM_LR = lr
    utils.OPTIM_BETAS = (beta1, beta2)
    utils.OPTIM_WEIGHT_DECAY = weight_decay

    model = LSTNET(first_domain, second_domain, params=fin_layer_params, optim_name=optimizer_name)

    train.MAX_PATIENCE = patience
    trained_model, val_loss = train.train_and_validate(model, train_loader, epochs, val_loader, run_optuna=True, trial=trial)

    trial.set_user_attr("best_model", trained_model)
    return val_loss


if __name__ == "__main__":
    utils.assign_device()
    utils.NUM_WORKERS = 8
    utils.OUTPUT_FOLDER = "output/"
    utils.BATCH_SIZE = 64
    
    utils.MANUAL_SEED = 42
    utils.VAL_SIZE = 0.25
    args = parse_args()
    
    train_loader, val_loader = data_preparation.get_training_loader(args.first_domain, args.second_domain, args.supervised, split_data=True)
    utils.PARAMS_FILE_PATH = args.params_file
    orig_layer_params = utils.get_networks_params()
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=20, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args.first_domain, args.second_domain, orig_layer_params, val_loader, train_loader), n_trials=100)

    trial = study.best_trial
    best_val_loss = trial.value
    best_params = trial.params

    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best params:')
    for key, value in best_params.items():
        print(f"\t{key}: {value}")

    output_dir = "optuna_lstnet/"
    os.makedirs(output_dir, exist_ok=True)

    best_model = study.best_trial.user_attrs["best_model"]
    torch.save(best_model, f"{output_dir}/model.pth")

    best_params["best_val_loss"] = best_val_loss
    with open(f"{output_dir}/params.json", "w") as file:
        json.dump(best_params, file, indent=2)

        # visualize importances and accuracies
    fig1 = plot_param_importances(study).figure
    fig1.savefig(f"{output_dir}/param_importances.png")
    plt.close(fig1)

    fig2 = plot_optimization_history(study).figure
    fig2.savefig(f"{output_dir}/optimization_history.png")
    plt.close(fig2)

    fig3 = plot_parallel_coordinate(study).figure
    fig3.savefig(f"{output_dir}/parallel_coordinate.png")
    plt.close(fig3)
