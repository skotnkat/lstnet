import copy

from data_preparation import AugmentOps
from LstnetTrainer import TrainParams


def suggest_weights(trial, weights_sum):
    tied = trial.suggest_categorical("tied", [True, False])

    # How much of the weight is given to the cycle consistency losses
    # -> domain + latent have the rest
    cycle_overall_share = trial.suggest_float("cycle_overall_share", 0.2, 0.95)
    rest_overall_share = 1.0 - cycle_overall_share

    # How much of the rest is given to the latent loss
    within_rest_latent_share = trial.suggest_float("within_rest_latent_share", 0.2, 0.8)
    latent_overall_share = rest_overall_share * within_rest_latent_share

    both_domain_overall_share = rest_overall_share - latent_overall_share

    if tied:
        d1_overall_share = both_domain_overall_share / 2.0
        d2_overall_share = both_domain_overall_share / 2.0

    else:
        d1_higher = trial.suggest_categorical("first_domain_higher", [True, False])
        domain_bias = trial.suggest_float("domain_bias", 0.55, 0.75)

        d1_within_domain_share = domain_bias if d1_higher else (1.0 - domain_bias)

        d1_overall_share = both_domain_overall_share * d1_within_domain_share
        d2_overall_share = both_domain_overall_share - d1_overall_share

    # How much of the cycle consistency weight is given to the full vs half cycle
    within_cycle_fc_share = trial.suggest_float(
        "within_cycle_full_cycle_share", 0.2, 0.8
    )
    fc_both_overall_share = cycle_overall_share * within_cycle_fc_share
    hc_both_overall_share = cycle_overall_share - fc_both_overall_share

    if tied:
        hc1_overall_share = hc_both_overall_share / 2.0
        hc2_overall_share = hc_both_overall_share / 2.0

        fc1_overall_share = fc_both_overall_share / 2.0
        fc2_overall_share = fc_both_overall_share / 2.0

    else:
        hc1_higher = trial.suggest_categorical("first_half_cycle_higher", [True, False])
        hc_bias = trial.suggest_float("half_cycle_bias", 0.55, 0.75)

        hc1_within_hc_share = hc_bias if hc1_higher else (1.0 - hc_bias)

        hc1_overall_share = hc_both_overall_share * hc1_within_hc_share
        hc2_overall_share = hc_both_overall_share - hc1_overall_share

        first_full_cycle_higher = trial.suggest_categorical(
            "first_full_cycle_higher", [True, False]
        )
        fc_bias = trial.suggest_float("full_cycle_bias", 0.55, 0.75)

        fc1_within_fc_share = fc_bias if first_full_cycle_higher else (1.0 - fc_bias)

        fc1_overall_share = fc_both_overall_share * fc1_within_fc_share
        fc2_overall_share = fc_both_overall_share - fc1_overall_share

    weights = [
        d1_overall_share,
        d2_overall_share,
        latent_overall_share,
        hc1_overall_share,
        hc2_overall_share,
        fc1_overall_share,
        fc2_overall_share,
    ]

    # Normalize weights to match original sum
    current_sum = sum(weights)
    norm_weights = [w * weights_sum / current_sum for w in weights]

    return norm_weights


def suggest_weights_reduced(trial, weights_sum):
    # originally: 20+20+30+4*100 = 470 -> 400 / 470 = 0.851
    cycle_overall_share = trial.suggest_float("cycle_overall_share", 0.75, 0.95)
    rest_overall_share = 1.0 - cycle_overall_share

    # How much of the rest is given to the latent loss
    within_rest_latent_share = trial.suggest_float("within_rest_latent_share", 0.2, 0.8)
    latent_overall_share = rest_overall_share * within_rest_latent_share

    both_domain_overall_share = rest_overall_share - latent_overall_share

    # How much of the cycle consistency weight is given to the full vs half cycle
    within_cycle_fc_share = trial.suggest_float(
        "within_cycle_full_cycle_share", 0.2, 0.8
    )
    fc_both_overall_share = cycle_overall_share * within_cycle_fc_share
    hc_both_overall_share = cycle_overall_share - fc_both_overall_share

    weights = [
        both_domain_overall_share / 2,
        both_domain_overall_share / 2,
        latent_overall_share,
        hc_both_overall_share / 2,
        hc_both_overall_share / 2,
        fc_both_overall_share / 2,
        fc_both_overall_share / 2,
    ]

    # Normalize weights to match original sum
    current_sum = sum(weights)
    norm_weights = [w * weights_sum / current_sum for w in weights]

    return norm_weights


def suggest_augment_params(trial):
    rotation = trial.suggest_int("rotation", 0, 30, step=5)
    zoom = trial.suggest_float("zoom", 0.0, 0.3, step=0.05)
    shift = trial.suggest_int("shift", 0, 5)

    return AugmentOps(rotation=rotation, zoom=zoom, shift=shift)


def suggest_training_params(trial, cmd_args):
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optim_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "Lion"])
    beta1 = trial.suggest_float("beta1", 0.8, 0.99)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)

    betas = (beta1, beta2)

    epoch_num = trial.suggest_int("max_epoch_num", 50, 200, step=50)

    return TrainParams(
        max_epoch_num=epoch_num,
        optim_name=optim_name,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        max_patience=cmd_args.patience,
    )


def suggest_architecture(trial, base_params):
    """
    Suggest architecture modifications with symmetry:
    - Domain encoders/generators modified together
    - Shared encoders/generators modified together
    - Domain discriminators modified together
    - Latent discriminator modified separately
    """

    params = copy.deepcopy(base_params)

    # Suggest extra layers for components
    domain_extra_layer = trial.suggest_categorical("domain_extra_layer", [True, False])
    shared_extra_layer = trial.suggest_categorical("shared_extra_layer", [True, False])
    domain_disc_extra_layer = trial.suggest_categorical(
        "domain_disc_extra_layer", [True, False]
    )
    latent_disc_extra_layer = trial.suggest_categorical(
        "latent_disc_extra_layer", [True, False]
    )

    # Suggest base output channels (will be doubled for each subsequent layer)
    # Base for first layers of encoders, rest will be derived
    base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])

    # Domain Encoders
    domain_encoders_layers_num = len(params["first_encoder"])

    for i in range(domain_encoders_layers_num):
        out_channels = base_channels * (2**i)
        params["first_encoder"][i]["out_channels"] = out_channels
        params["second_encoder"][i]["out_channels"] = out_channels

    last_domain_encoder_out_channels = base_channels * (
        2 ** (domain_encoders_layers_num - 1)
    )
    if domain_extra_layer:
        last_domain_encoder_out_channels = base_channels * (
            2**domain_encoders_layers_num
        )

        extra_layer = {
            "out_channels": last_domain_encoder_out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        params["first_encoder"].append(extra_layer)
        params["second_encoder"].append(extra_layer)

    # Shared Encoder
    for i in range(len(params["shared_encoder"])):
        out_channels = last_domain_encoder_out_channels / (1 + 2**i)
        params["shared_encoder"][i]["out_channels"] = out_channels

    shared_encoder_last_out_channels = last_domain_encoder_out_channels / (
        2 ** (len(params["shared_encoder"]) - 1)
    )
    if shared_extra_layer:
        shared_encoder_last_out_channels = last_domain_encoder_out_channels / (
            2 ** len(params["shared_encoder"])
        )
        extra_layer = {
            "out_channels": shared_encoder_last_out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        params["shared_encoder"].append(extra_layer)

    # Latent Discriminator
    latent_disc_fin_layers = len(params["latent_discriminator"]) + int(
        latent_disc_extra_layer
    )
    first_half_layers = latent_disc_fin_layers // 2 + 1
    even_layers_flag = True if latent_disc_fin_layers % 2 == 0 else False

    out_channels = shared_encoder_last_out_channels
    for i in range(first_half_layers):
        out_channels = shared_encoder_last_out_channels * 2
        params["latent_discriminator"][i]["out_channels"] = out_channels

    if even_layers_flag:
        new_layer = {
            "out_channels": out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        params["latent_discriminator"].insert(first_half_layers, new_layer)

        # Reduce for second half
        for i in range(first_half_layers + 1, len(params["latent_discriminator"])):
            reduction_factor = 2 ** (i - first_half_layers)
            out_channels = (
                shared_encoder_last_out_channels
                * (2**first_half_layers)
                / reduction_factor
            )
            params["latent_discriminator"][i]["out_channels"] = out_channels

    # Shared Generator
    shared_generator_layers_num = len(params["shared_generator"])
    for i in range(shared_generator_layers_num):
        out_channels = shared_encoder_last_out_channels * (2**i)
        params["shared_generator"][i]["out_channels"] = out_channels

    shared_generator_last_out_channels = shared_encoder_last_out_channels * (
        2**shared_generator_layers_num
    )
    if shared_extra_layer:
        shared_generator_last_out_channels = shared_encoder_last_out_channels * (
            2**shared_generator_layers_num
        )
        extra_layer = {
            "out_channels": shared_generator_last_out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        params["shared_generator"].append(extra_layer)

    # Domain Generator
    domain_generators_layers_num = len(params["first_generator"])

    first_domain_generator_out_channels = shared_encoder_last_out_channels * 2

    for i in range(domain_generators_layers_num):
        out_channels = first_domain_generator_out_channels * (
            2 ** (domain_generators_layers_num - i - 1)
        )
        params["first_generator"][i]["out_channels"] = out_channels
        params["second_generator"][i]["out_channels"] = out_channels

    if domain_extra_layer:
        extra_layer = {
            "out_channels": first_domain_generator_out_channels
            / (2 ** (domain_generators_layers_num - i)),
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        params["first_generator"].append(extra_layer)
        params["second_generator"].append(extra_layer)

    # Domain Discriminators
    domain_disc_layers_num = len(params["first_discriminator"])
    for i in range(domain_disc_layers_num):
        out_channels = base_channels * (2**i)
        params["first_discriminator"][i]["out_channels"] = out_channels
        params["second_discriminator"][i]["out_channels"] = out_channels

    if domain_disc_extra_layer:
        out_channels = base_channels * (2**domain_disc_layers_num)
        extra_conv = {
            "out_channels": out_channels,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }

        extra_max_pool = {
            "kernel_size": 2,
            "stride": 2,
            "padding": "same",
        }

        params["first_discriminator"].append([extra_conv, extra_max_pool])
        params["second_discriminator"].append([extra_conv, extra_max_pool])

    print(params)
    return params
