import argparse
import os
import json

import utils
import train
import domain_adaptation
import torch
import data_preparation
from models.lstnet import LSTNET
from eval_models.clf_models import MnistClf, UspsClf, SvhnClf


def add_common_args(parser):
    parser.add_argument("--output_folder", type=str, default="output/", help="Path to the output folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of batches used in training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Size of batches used in training.")
    parser.add_argument("--load_model", action="store_true",
                        help="If a model with name 'model_name' should be loaded for data translation.")
    parser.add_argument("--manual_seed", type=int, default=42)

    return parser


def add_train_args(parser):
    parser.add_argument("first_domain", type=str.upper, help="Name of the first dataset.")
    parser.add_argument("second_domain", type=str.upper, help="Name of the second dataset.")
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")

    parser.add_argument("--supervised", action="store_true",
                        help="Run supervised domain adaptation. If not set, unsupervised domain adaptation is run.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate used in Adam optimizer.")
    parser.add_argument("--decay", type=float, nargs=2, default=(0.8, 0.999),
                        help="Two float values for Adam optimizer decay (beta1, beta2)")
    parser.add_argument("--full_training_only", action="store_true",
                        help="If after train and validate another round of training should be run with full training set.")

    parser.add_argument("--epoch_num", type=int, default=50)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

def add_translate_args(parser):
    parser.add_argument("domain", type=str.upper, help="Name of the domain to be translated to the other domain.")
    parser.add_argument("--model_name", type=str, default="lstnet.pth",
                        help="Name of the model to be loaded for translation")

def add_eval_args(parser):
    parser.add_argument("domain", type=str.upper, help="Name of the domain to be evaluated.")
    parser.add_argument("clf_model", type=str, help="Name of the model to classify the data.")
    parser.add_argument("--dataset_path", default="", type=str, help="Name of file to load the dataset from")
    parser.add_argument("--output_results_file", default="results_json", type=str)


def add_end_to_end_parser(parser):
    add_train_args(parser)

    parser.add_argument("clf_first_domain", type=str, help="Path to the trained classifier of the first domain")
    parser.add_argument("clf_second_domain", type=str, help="Path to the trained classifier of the second domain")
    parser.add_argument("--save_trans_data", action="store_true",
                        help="Bool if the translated data that are result of the translation phase should be saved in files")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Domain adaptation: train, translate, evalute or all"
    )

    subparsers = parser.add_subparsers(dest="operation", required=True)

    # Train subparser
    train_parser = subparsers.add_parser("train", help="Train the LSTNET model for domain adaptation task.")
    add_common_args(train_parser)
    add_train_args(train_parser)

    # Translation
    trans_parser = subparsers.add_parser("translate",
                                         help="Load a trained model and dataset and map the images from the original domain to the second one.")
    add_common_args(trans_parser)
    add_translate_args(trans_parser)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Load a dataset and predict their labels using a given classifier")
    add_common_args(eval_parser)
    add_eval_args(eval_parser)

    # All: train->translate->evaluate
    all_parser = subparsers.add_parser("all",
                                       help="Perform the end-to-end workflow. Train the model. Load the test datasets and translate it to their respective target domains. Load a classifier and predict labels for the translated images.")
    add_common_args(all_parser)
    add_end_to_end_parser(all_parser)

    args = parser.parse_args()

    utils.OUTPUT_FOLDER = utils.check_file_ending(args.output_folder, '/')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)


    if args.operation == ['eval', 'all']:
        args.clf_model = utils.check_file_ending(args.clf_model, '.pth')


    return args


def initialize(args):
    utils.NUM_WORKERS = args.num_workers
    utils.BATCH_SIZE = args.batch_size
    utils.MANUAL_SEED = args.manual_seed
    utils.VAL_SIZE = args.val_size

    utils.PARAMS_FILE_PATH = "mnist_usps_params.json"
    if args.operation in ['train', 'all']:
        utils.PARAMS_FILE_PATH = args.params_file #  svae params in architecture
        utils.ADAM_LR = args.learning_rate
        utils.ADAM_DECAY = args.decay
        train.MAX_PATIENCE = args.epoch_num  # by default no early stopping (number of epochs)

        if args.early_stopping:
            train.MAX_PATIENCE = args.patience


    utils.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Device being used: {utils.DEVICE}')


def run_training(args, return_model=False):
    if not args.full_training_only:
        model = train.run(args.first_domain, args.second_domain, args.supervised, args.epoch_num)

    else:
        model = train.run_full_training(args.first_domain, args.second_domain, args.supervised, args.epoch_num)

    if return_model:
        return model


def run_translation(args, domain, model=None, return_data=False, save_trans_data=True):
    if model is None and args.load_model is False:
        raise ValueError("Model for translation is not specified.")

    if args.load_model:
        model = LSTNET.load_lstnet_model(f'{utils.OUTPUT_FOLDER}{args.model_name}')

    translated_data = domain_adaptation.adapt_domain(model, domain)

    if save_trans_data:
        file_name = f"{domain}_translated_data.pt"
        torch.save(translated_data, f'{utils.OUTPUT_FOLDER}{file_name}')

    if return_data:
        return translated_data


def run_evaluation(clf_name, domain_name, data_path=""):
    model = torch.load(clf_name, weights_only=False, map_location=utils.DEVICE)
    test_acc = domain_adaptation.evaluate(model, domain_name, data_path)

    with open(f'{utils.OUTPUT_FOLDER}{domain_name}_eval_results.json', 'w') as file:
        json.dump({f'test_acc': test_acc}, file, indent=2)


def run_end_to_end(args):
    model = run_training(args, return_model=True)

    run_translation(args, args.first_domain, model, args.save_trans_data)  # might be better to return data and immediately use them in eval
    run_translation(args, args.second_domain, model, args.save_trans_data)

    run_evaluation(args.clf_second_domain, args.first_domain)
    run_evaluation(args.clf_first_domain, args.second_domain)


if __name__ == "__main__":
    args = parse_args()
    initialize(args)

    if args.operation == 'train':
        run_training(args)

    elif args.operation == 'translate':
        run_translation(args, args.domain, save_trans_data=True)

    elif args.operation == 'eval':
        run_evaluation(args.clf_model, args.domain, args.dataset_path)

    else:
        run_end_to_end(args)
