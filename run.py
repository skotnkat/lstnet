import argparse
import os

import utils
from train import train
from predict import predict
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("first_domain", type=str, help="Name of the first dataset.")
    parser.add_argument("second_domain", type=str, help="Name of the second dataset.")
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")
    parser.add_argument("--operation", type=str, choices=["train", "predict", "all"], default="all",
                        help="Operation to perform: train, predict, or all (first training, then prediction).")
    parser.add_argument("--supervised", action="store_true",
                        help="Run supervised domain adaptation. If not set, unsupervised domain adaptation is run.")

    parser.add_argument("--output_folder", type=str, default="output", help="Path to the output folder")
    parser.add_argument("--model_name", type=str, default="lstnet_model", help="Name of the file to store the trained model.")
    parser.add_argument("--loss_file", type=str, default="loss", help="File with recorded losses for each epoch.")

    parser.add_argument("--num_workers", type=int, default=4, help="Size of batches used in training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of batches used in training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate used in Adam optimizer.")
    parser.add_argument("--decay", type=float, nargs=2, default=(0.9, 0.999), help="Two float values for Adam optimizer decay (beta1, beta2)")
    parser.add_argument("--delta_loss", type=float, default=1e-4, help="Maximum allowed change in loss between iterations to consider convergence")


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
