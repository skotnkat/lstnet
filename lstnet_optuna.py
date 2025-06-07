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
import pickle
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('first_domain', type=str.upper)
    parser.add_argument('second_domain', type=str.upper)
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--study_name", type=str, default="lstnet_db")

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

    extra_layer = trial.suggest_categorical("d_extra_layer", [True, False])
    base = 1024
    if extra_layer:
        kernel_size = trial.suggest_categorical("d_extra_layer_kernel_size", [2, 3, 5])
        extra_conv = get_stand_conv_params(base, kernel_size)
        max_pool_params = get_stand_max_pool_params(kernel_size)

        new_layer_params["first_discriminator"].insert(-1, [extra_conv, max_pool_params])
        new_layer_params["second_discriminator"].insert(-1, [extra_conv, max_pool_params])

    shared_layers_num = trial.suggest_categorical("d_shared_layers_num", [3, 5])
    kernel_size = trial.suggest_categorical("d_shared_kernel_size", [3, 5])

    latent_disc_params = []
    base = trial.suggest_categorical(f"shared_disc_base", [128, 256])
    half = shared_layers_num // 2
    up_channels = [base * (2**i) for i in range(half)]
    down_channels = list(reversed(up_channels[:-1]))  # drop repetition peak

    channels_sequence = up_channels + up_channels[-1] + down_channels

    for out_channels in channels_sequence:
        conv_params = get_stand_conv_params(out_channels, kernel_size)
        max_pool_params = get_stand_max_pool_params(kernel_size)

        latent_disc_params.append([conv_params, max_pool_params])

    # with out channels
    latent_disc_params.append(orig_layer_params["latent_discriminator"][-1])

    new_layer_params["latent_discriminator"] = latent_disc_params

    return new_layer_params


# update to be able to introduce asymetry (new function)
def update_enc_gen_params(trial, orig_layer_params):
    extra_layer = trial.suggest_categorical("eg_extra_layer", [True, False])

    new_layer_params = copy.deepcopy(orig_layer_params)

    if extra_layer:
        out_channels = new_layer_params["first_encoder"][-1]["out_channels"] * 2
        kernel_size = trial.suggest_categorical("eg_extra_layer_kernel_size", [3, 5, 7])
        extra_conv = get_stand_conv_params(out_channels, kernel_size)

        new_layer_params["first_encoder"].append(extra_conv)
        new_layer_params["second_encoder"].append(extra_conv)

        new_layer_params["first_generator"].insert(0, extra_conv)
        new_layer_params["second_generator"].insert(0, extra_conv)


    # shared encoder generator
    shared_layers_num = trial.suggest_int("eg_shared_layers_num", 3, 5)
    kernel_size = trial.suggest_categorical("eg_shared_kernel_size", [3, 5])

    base = trial.suggest_categorical("eg_shared_base_out_channels", [256, 512])

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


def objective(trial, first_domain, second_domain, orig_layer_params, val_loader, train_loader, max_epochs):
    start_time = time.time()
    updated_layer_params = update_enc_gen_params(trial, orig_layer_params)
    fin_layer_params = update_disc_params(trial, updated_layer_params)

    leaky_relu_neg_slope = trial.suggest_float("negative_slope", 0.01, 0.3)
    batch_norm_momentum = trial.suggest_float("momentum", 0.01, 0.3)
    fin_layer_params["leaky_relu"] = {"negative_slope": leaky_relu_neg_slope}
    fin_layer_params["batch_norm"] = {"momentum": batch_norm_momentum}

    tie_weights = trial.suggest_categorical("tie_sharing_domain_weights", [True, False])
    if tie_weights:
        an_weights = trial.suggest_int("w_an", 20, 100, step=20)  # w1, w2
        w_l = trial.suggest_int("w_l", 20, 100, step=20)  # wl
        w_cc = trial.suggest_int("w_cc", 20, 100, step=20)  # w_3, w_4
        w_full_cc = trial.suggest_int("w_full_cc", 20, 100, step=20)  # w_4, w_5

        w_raw = [an_weights, an_weights, w_cc, w_cc, w_full_cc, w_full_cc, w_l]
    else:
        weights_num = 7
        w_raw = [trial.suggest_int(f'w_{i}', 20, 100, step=20) for i in range(1, weights_num+1)]

    w_sum = sum(w_raw)
    # normalize weights for comparison
    weights = [w / w_sum * 100 for w in w_raw]

    loss_functions.W_1 = weights[0]
    loss_functions.W_2 = weights[1]
    loss_functions.W_3 = weights[2]
    loss_functions.W_4 = weights[3]
    loss_functions.W_5 = weights[4]
    loss_functions.W_6 = weights[5]
    loss_functions.W_l = weights[6]
    
    epoch_num = trial.suggest_int("epoch_num", 25, 150, step=25)
    patience = trial.suggest_int("patience", 5, 20, step=5)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"])

    beta1 = trial.suggest_float("beta1", 0.85, 0.95)
    beta2 = trial.suggest_float("beta2", 0.98, 0.999)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-4, 1e-2])

    utils.OPTIM_LR = lr
    utils.OPTIM_BETAS = (beta1, beta2)
    utils.OPTIM_WEIGHT_DECAY = weight_decay

    model = LSTNET(first_domain, second_domain, params=fin_layer_params, optim_name=optimizer_name)

    train.MAX_PATIENCE = patience
    trained_model, val_loss, best_epoch_idx, last_epoch_idx = train.train_and_validate(model, train_loader, epoch_num, val_loader, run_optuna=True, trial=trial)

    model.to("cpu")
    trained_model.to("cpu")
    
    model_path = f"optuna_lstnet/model_trial_{trial.number}.pth"
    trained_model.save_model(model_path)
    trial.set_user_attr("model_path", model_path)
    trial.set_user_attr("best_epoch_idx", best_epoch_idx)
    trial.set_user_attr("last_run_epoch_idx", last_epoch_idx)

    del model, trained_model
    torch.cuda.empty_cache()

    end_time = time.time()

    print(f'Trial took: {(end_time-start_time)/3600} hours')
    
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

    output_dir = "optuna_lstnet"
    os.makedirs(f"{output_dir}", exist_ok=True)

    max_epochs = 150
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, interval_steps=1)
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", 
                                study_name=args.study_name, load_if_exists=True, storage=f"sqlite:///{args.study_name}.db", 
                                sampler=sampler, pruner=pruner)
    
    study.optimize(lambda trial: objective(trial, args.first_domain, args.second_domain, orig_layer_params, val_loader, train_loader, max_epochs),
                   n_trials=200,
                   show_progress_bar=True, 
                   gc_after_trial=True)

    trial = study.best_trial
    best_val_loss = trial.value
    best_params = trial.params

    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best params:')
    for key, value in best_params.items():
        print(f"\t{key}: {value}")

    best_params["best_val_loss"] = best_val_loss
    with open(f"{output_dir}/params.json", "w") as file:
        json.dump(best_params, file, indent=2)
        
    best_model_path = study.best_trial.user_attrs["model_path"]

    
    with open(f"optuna_lstnet/{args.study_name}.pkl", "wb") as f:
        pickle.dump(study, f)

    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        
        # Skip if it's the best model
        if os.path.abspath(file_path) == os.path.abspath(best_model_path):
            continue

        # Remove only .pth files
        if file_path.endswith(".pth") and os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")