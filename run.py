"""
Main script to run training, translation and evaluation
for img-to-img translation using LSTNET model.
"""

import os
import argparse
import json
from typing import Optional
import torch
from torch.utils.data import TensorDataset


import utils
import train
import domain_adaptation
from data_preparation import AugmentOps
from LstnetTrainer import TrainParams

from models.lstnet import LSTNET
import lstnet_optuna


# TODO: remove the --compile logic (makes it too complex for users -> drop last causes issue with comparability)

def add_common_args(parser: argparse.ArgumentParser):
    """Add arguments common to all operations."""
    _ = parser.add_argument(
        "--output_folder", type=str, default="output/", help="Path to the output folder"
    )
    _ = parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of batches used in training."
    )
    _ = parser.add_argument(
        "--num_workers", type=int, default=4, help="Size of batches used in training."
    )
    _ = parser.add_argument(
        "--load_model",
        action="store_true",
        help="If a model with name 'model_name' should be loaded for data translation.",
    )
    _ = parser.add_argument("--manual_seed", type=int, default=42)


def add_train_args(parser: argparse.ArgumentParser):
    """Add arguments specific for training operation."""
    _ = parser.add_argument(
        "first_domain", type=str.upper, help="Name of the first dataset."
    )
    _ = parser.add_argument(
        "second_domain", type=str.upper, help="Name of the second dataset."
    )
    _ = parser.add_argument(
        "params_file",
        type=str,
        help="Path to the file with stored parameters of model.",
    )

    _ = parser.add_argument(
        "--supervised",
        action="store_true",
        help="Run supervised domain adaptation. If not set, unsupervised domain adaptation is run.",
    )

    _ = parser.add_argument("--optim_name", type=str, default="Adam")

    _ = parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate used in Adam optimizer.",
    )
    _ = parser.add_argument(
        "--betas",
        type=float,
        nargs=2,
        default=(0.8, 0.999),
        help="Two float values for Adam optimizer decay (beta1, beta2)",
    )
    _ = parser.add_argument("--weight_decay", type=float, default=1e-2)
    _ = parser.add_argument(
        "--full_training_only",
        action="store_true",
        help="If set, the full training set will be used. No validation phase after training.",
    )

    _ = parser.add_argument("--model_file_name", type=str, default="lstnet.pth")
    _ = parser.add_argument("--logs_file_name", type=str, default="loss_logs.json")

    _ = parser.add_argument("--epoch_num", type=int, default=50)
    _ = parser.add_argument("--val_size", type=float, default=0.25)
    _ = parser.add_argument("--early_stopping", action="store_true")
    _ = parser.add_argument("--patience", type=int, default=10)
    _ = parser.add_argument("--rotation", type=int, default=10)
    _ = parser.add_argument("--zoom", type=float, default=0.1)
    _ = parser.add_argument("--shift", type=int, default=2)
    _ = parser.add_argument("--skip_augmentation", action="store_true")
    _ = parser.add_argument("--use_svhn_extra", action="store_true")

    _ = parser.add_argument(
        "--weights",
        type=float,
        nargs=7,
        default=[20, 20, 30, 100, 100, 100, 100],
        help="List of 7 float weights",
    )

    _ = parser.add_argument("--resize", type=int, nargs=2, default=None)

    _ = parser.add_argument(
        "--compile",
        action="store_true",
        help="If set, the model will be compiled before training (pytorch compile).",
    )
    _ = parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="If set, gradient checkpointing will be enabled to reduce GPU memory usage.",
    )

    _ = parser.add_argument("--optuna", action="store_true")
    _ = parser.add_argument("--optuna_study_name", type=str, default="lstnet_study")
    _ = parser.add_argument(
        "--optuna_trials",
        type=int,
        default=50,
        help="Number of Optuna trials to perform if --optuna is set.",
    )
    # _ = parser.add_argument("--optuna_max_resource", type=int, default=20)
    # _ = parser.add_argument("--optuna_min_resource", type=int, default=5)
    # _ = parser.add_argument("--optuna_reduction_factor", type=int, default=2)
    _ = parser.add_argument("--optuna_sampler_start_trials", type=int, default=20)
    _ = parser.add_argument("--optuna_pruner_sample_trials", type=int, default=50)
    _ = parser.add_argument("--optuna_pruner_warmup_steps", type=int, default=15)
    _ = parser.add_argument("--optuna_pruner_interval_steps", type=int, default=5)
    _ = parser.add_argument("--percentile", type=int, default=10)
    _ = parser.add_argument(
        "--hyperparam_mode",
        type=str,
        nargs="*",
        choices=[
            "weights",
            "weights_reduced",
            "augm_ops",
            "train_params",
            "architecture",
        ],
        default=[],
        help=(
            "List of hyperparameter modes to run (choose zero or more from: "
            "weights, weights_reduced, augm_ops, train_params, architecture)."
        ),
    )


def add_translate_args(parser: argparse.ArgumentParser):
    """Add arguments specific for translation operation."""
    _ = parser.add_argument(
        "domain",
        type=str.upper,
        help="Name of the domain to be translated to the other domain.",
    )
    _ = parser.add_argument(
        "--model_name",
        type=str,
        default="lstnet.pth",
        help="Name of the model to be loaded for translation",
    )


def add_eval_args(parser: argparse.ArgumentParser):
    """Add arguments specific for evaluation operation."""
    _ = parser.add_argument(
        "domain", type=str.upper, help="Name of the domain to be evaluated."
    )
    _ = parser.add_argument(
        "clf_model", type=str, help="Name of the model to classify the data."
    )
    _ = parser.add_argument(
        "--dataset_path",
        default="",
        type=str,
        help="Name of file to load the dataset from",
    )
    _ = parser.add_argument("--output_results_file", default="results_json", type=str)
    _ = parser.add_argument("--log_name", default="test_acc", type=str)


def add_end_to_end_parser(parser: argparse.ArgumentParser):
    """Combine arguments (train, translate, evaluate) for end-to-end operation."""
    add_train_args(parser)

    _ = parser.add_argument(
        "clf_first_domain",
        type=str,
        help="Path to the trained classifier of the first domain",
    )
    _ = parser.add_argument(
        "clf_second_domain",
        type=str,
        help="Path to the trained classifier of the second domain",
    )
    _ = parser.add_argument(
        "--save_trans_data",
        action="store_true",
        help="If set, the translated data should be saved.",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Domain adaptation: train, translate, evalute or all"
    )

    subparsers = parser.add_subparsers(dest="operation", required=True)

    # Train subparser
    train_parser = subparsers.add_parser(
        "train", help="Train the LSTNET model for domain adaptation task."
    )
    add_common_args(train_parser)
    add_train_args(train_parser)

    # Translation
    trans_parser = subparsers.add_parser(
        "translate",
        help="Load a trained LSTNET model \
            and translate images from the source domain to the target domain.",
    )
    add_common_args(trans_parser)
    add_translate_args(trans_parser)

    # Eval
    eval_parser = subparsers.add_parser(
        "eval", help="Load a dataset and predict their labels using a given classifier"
    )
    add_common_args(eval_parser)
    add_eval_args(eval_parser)

    # All: train->translate->evaluate
    all_parser = subparsers.add_parser(
        "all",
        help="Perform the end-to-end workflow.",
    )
    add_common_args(all_parser)
    add_end_to_end_parser(all_parser)

    cmd_args = parser.parse_args()

    if not os.path.exists(cmd_args.output_folder):
        os.makedirs(cmd_args.output_folder)

    if cmd_args.operation in ["eval", "all"]:
        cmd_args.clf_model = utils.check_file_ending(cmd_args.clf_model, ".pth")

    return cmd_args


def run_training(
    cmd_args: argparse.Namespace, return_model: bool = False
) -> Optional[LSTNET]:
    """
    Run training for the LSTNET model.
    Saves the trained model and training logs.

    Trained model to a file in cmd_args.output_folder with name cmd_args.model_name.
    Training logs to a file in cmd_args.output_folder with name cmd_args.log_name.


    Args:
        cmd_args (argparse.Namespace): Command line arguments.
        return_model (bool, optional): Whether to return the trained model.
            Defaults to False.

    Returns:
        Optional[torch.nn.Module]: If return_model is True, then the trained LSTNET model.
            Otherwise None.
    """
    if cmd_args.optuna:
        print("Running Optuna hyperparameter optimization for LSTNET model.")
        model = lstnet_optuna.run_optuna_lstnet(cmd_args)

        if return_model:
            return model

        return

    run_validation_flag = not cmd_args.full_training_only

    with open(f"{cmd_args.params_file}", "r", encoding="utf-8") as file:
        params = json.load(file)

    # Create TrainParams object from args
    max_patience = cmd_args.patience if cmd_args.early_stopping else None

    train_params = TrainParams(
        max_epoch_num=cmd_args.epoch_num,
        max_patience=max_patience,
        optim_name=cmd_args.optim_name,
        lr=cmd_args.learning_rate,
        betas=tuple(cmd_args.betas),
        weight_decay=cmd_args.weight_decay,
    )

    # Create AugmentOps object from args
    augm_ops = AugmentOps(
        rotation=cmd_args.rotation, zoom=cmd_args.zoom, shift=cmd_args.shift
    )

    model = train.run(
        cmd_args.first_domain,
        cmd_args.second_domain,
        supervised=cmd_args.supervised,
        params=params,
        weights=cmd_args.weights,
        run_validation=run_validation_flag,
        val_data_size=cmd_args.val_size,
        batch_size=cmd_args.batch_size,
        num_workers=cmd_args.num_workers,
        output_folder=cmd_args.output_folder,
        model_file_name=cmd_args.model_file_name,
        logs_file_name=cmd_args.logs_file_name,
        manual_seed=cmd_args.manual_seed,
        augm_ops=augm_ops,
        skip_augmentation=cmd_args.skip_augmentation,
        resize=cmd_args.resize,
        train_params=train_params,
        compile_model=cmd_args.compile,
        use_checkpoint=cmd_args.use_checkpoint,
        use_svhn_extra=cmd_args.use_svhn_extra,
    )

    if return_model:
        return model


def run_translation(
    cmd_args: argparse.Namespace,
    domain: str,
    model: Optional[LSTNET] = None,
    return_data: bool = False,
    save_trans_data: bool = True,
) -> Optional[TensorDataset]:
    """
    Run translation for a provided domain.
    Translate them from original domain (given by `domain` argument)
    into the the other domain (saved in the model itself).

    Args:
        cmd_args (argparse.Namespace): Command line arguments.
        domain (str): Name of the original domain (to be translated to the other domain).
        model (LSTNET, optional): The LSTNET model to use for translation.
            Defaults to None. If not provided and cmd_args.load_model is True,
            it will be loaded from file.
            The path the model is constructed from cmd_args.output_folder and cmd_args.model_name.
        return_data (bool, optional): Whether to return the translated data.
            Defaults to False.
        save_trans_data (bool, optional): Whether to save the translated data to file.
            Defaults to True.

    Raises:
        ValueError: If model is not specified and cmd_args.load_model is False.

    Returns:
        Optional[TensorDataset]: The translated dataset if return_data is True, None otherwise.
    """
    if model is None and cmd_args.load_model is False:
        raise ValueError("Model for translation is not specified.")

    if cmd_args.load_model:
        model = LSTNET.load_lstnet_model(
            f"{cmd_args.output_folder}/{cmd_args.model_name}"
        )

    translated_data = domain_adaptation.adapt_domain(
        model, domain, batch_size=cmd_args.batch_size, num_workers=cmd_args.num_workers
    )

    if save_trans_data:
        file_name = f"{domain}_translated_data.pt"
        torch.save(translated_data, f"{cmd_args.output_folder}/{file_name}")

    if return_data:
        return translated_data


def run_evaluation(
    cmd_args: argparse.Namespace,
    clf_name: str,
    domain_name: str,
    log_name: str,
    data_path: str = "",
    translated_data: Optional[TensorDataset] = None,
) -> None:
    """
    Run evaluation for a provided domain.
    Saves the evaluation results to a file
    in cmd_args.output_folder with name cmd_args.output_results_file.

    Args:
        cmd_args (argparse.Namespace): Command line arguments.
        clf_name (str): Name of the classifier model.
        domain_name (str): Name of the domain to evaluate.
        log_name (str): Name of the log file.
        data_path (str, optional): Path to the dataset. Defaults to "".
        translated_data (Optional[torch.Tensor], optional):
            Translated data to use for evaluation. Defaults to None.
    """
    model = torch.load(clf_name, weights_only=False, map_location=utils.DEVICE)

    test_acc = domain_adaptation.evaluate(
        clf=model,
        orig_domain_name=domain_name,
        data_path=data_path,
        batch_size=cmd_args.batch_size,
        num_workers=cmd_args.num_workers,
        translated_data=translated_data,
    )

    results_file = f"{cmd_args.output_folder}/{domain_name}_eval_results.json"
    print(f'Results saved to "{results_file}"')
    with open(results_file, "a", encoding="utf-8") as file:
        json.dump({f"{log_name}": test_acc}, file, indent=2)


def run_end_to_end(cmd_args: argparse.Namespace) -> None:
    """Run end-to-end operation: train -> translate -> evaluate.

    Args:
        cmd_args (argparse.Namespace): Command line arguments.
    """
    if cmd_args.optuna:
        print("Running Optuna hyperparameter optimization for LSTNET model.")
        model = lstnet_optuna.run_optuna_lstnet(cmd_args)

    else:
        model = run_training(cmd_args, return_model=True)

    # Translate first and second domains
    first_translated_data = run_translation(
        cmd_args,
        cmd_args.first_domain,
        model,
        return_data=True,
        save_trans_data=cmd_args.save_trans_data,
    )

    second_translated_data = run_translation(
        cmd_args,
        cmd_args.second_domain,
        model,
        return_data=True,
        save_trans_data=cmd_args.save_trans_data,
    )

    # Always use in-memory translated data for efficiency
    # Files are saved by run_translation when save_trans_data=True
    run_evaluation(
        cmd_args,
        cmd_args.clf_second_domain,
        cmd_args.first_domain,
        cmd_args.log_name,
        translated_data=first_translated_data,
    )
    run_evaluation(
        cmd_args,
        cmd_args.clf_first_domain,
        cmd_args.second_domain,
        cmd_args.log_name,
        translated_data=second_translated_data,
    )


if __name__ == "__main__":
    args = parse_args()
    utils.init_device()
    print(f"Device being used: {utils.DEVICE}")

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    if args.operation == "train":
        _ = run_training(args)

    elif args.operation == "translate":
        _ = run_translation(args, args.domain, save_trans_data=True)

    elif args.operation == "eval":
        run_evaluation(
            args,
            args.clf_model,
            args.domain,
            args.log_name,
            args.dataset_path,
        )

    else:
        run_end_to_end(args)
