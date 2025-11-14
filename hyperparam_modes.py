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

    return TrainParams(
        max_epoch_num=cmd_args.epoch_num,
        optim_name=optim_name,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        max_patience=cmd_args.patience,
    )
