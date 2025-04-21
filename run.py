import argparse
import os

import utils
from train import train
from predict import predict
import torch


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str, choices=["train", "predict", "all"], default="all",
                               help="Operation to perform: train, predict, or all (first training, then prediction).")
    parser.add_argument("--output_folder", type=str, default="output", help="Path to the output folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of batches used in training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Size of batches used in training.")

    # change the name to lstnet_path?
    parser.add_argument("--model_name", type=str, default="lstnet_model",
                        help="Name of the file to store the trained model.")

    return parser


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("first_domain", type=str, help="Name of the first dataset.")
    parser.add_argument("second_domain", type=str, help="Name of the second dataset.")
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")

    parser.add_argument("--supervised", action="store_true",
                              help="Run supervised domain adaptation. If not set, unsupervised domain adaptation is run.")

    parser.add_argument("--loss_file", type=str, default="loss", help="File with recorded losses for each epoch.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate used in Adam optimizer.")
    parser.add_argument("--decay", type=float, nargs=2, default=(0.9, 0.999),
                              help="Two float values for Adam optimizer decay (beta1, beta2)")
    parser.add_argument("--delta_loss", type=float, default=1e-4,
                              help="Maximum allowed change in loss between iterations to consider convergence")

    return parser


def get_translate_parser():
    parser = argparse.ArgumentParser()

    return parser

def get_eval_parser():
    parser = argparse.ArgumentParser()

    return parser


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain adaptation: train, translate, evalute or all"
    )

    common_parser = get_common_parser()
    train_parser = get_train_parser()
    trans_parser = get_translate_parser()
    eval_parser = get_eval_parser()

    subparsers = parser.add_subparsers(
        dest="operation",
        required=True,
        help="Operation to perform"
    )

    # Train
    p_train = subparsers.add_parser(
        "train",
        parents=[common_parser, train_parser],
        help="Train the LSTNET model for domain adaptation task."
    )

    # Translation
    subparsers.add_parser(
        "translate",
        parents=[common_parser, trans_parser],
        help="Load a trained model and dataset and map the images from the original domain to the second one."
    )

    # Eval
    subparsers.add_parser(
        "eval",
        parents=[common_parser, eval_parser],
        help="Load a dataset and predict their labels using a given classifier"
    )

    # All: train->translate->evaluate
    subparsers.add_parser(
        "all",
        parents=[common_parser, train_parser, trans_parser, eval_parser],
        help="Perform the end-to-end workflow. Train the model. Load the test datasets and translate it \
        to their respective target domains. Load a classifier and predict labels for the translated images."
    )

    args = parser.parse_args()
    return args


def initialize(args):
    utils.PARAMS_FILE_PATH = args.params_file
    utils.LOSS_FILE = args.loss_file
    utils.OUTPUT_FOLDER = args.output_folder

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    utils.NUM_WORKERS = args.num_workers
    utils.BATCH_SIZE = args.batch_size
    utils.ADAM_LR = args.learning_rate
    utils.ADAM_DECAY = args.decay
    utils.DELTA_LOSS = args.delta_loss

    utils.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device being used: {utils.DEVICE}')


def main():
    args = parse_args()
    initialize(args)

    model = None
    model_path = f'{utils.OUTPUT_FOLDER}/{args.model_name}.pth'

    first_domain = args.first_domain.upper()
    second_domain = args.second_domain.upper()

    if (args.operation == "train") or (args.operation == "all"):
        model = train(first_domain, second_domain, args.supervised)
        torch.save(model, model_path)

    if (args.operation == "predict") or (args.operation == "all"):
        if model is None:
            model = torch.load(model_path)

        scores = predict(model)


if __name__ == "__main__":
    main()
