import argparse
import os
import json

import utils
import train
import domain_adaptation
import torch


def add_common_args(parser):
    parser.add_argument("--output_folder", type=str, default="output/", help="Path to the output folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of batches used in training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Size of batches used in training.")

    return parser


def add_train_args(parser):
    parser.add_argument("first_domain", type=str.upper, help="Name of the first dataset.")
    parser.add_argument("second_domain", type=str.upper, help="Name of the second dataset.")
    parser.add_argument("params_file", type=str, help="Path to the file with stored parameters of model.")

    parser.add_argument("--supervised", action="store_true",
                        help="Run supervised domain adaptation. If not set, unsupervised domain adaptation is run.")

    parser.add_argument("--loss_file", type=str, default="loss", help="File with recorded losses for each epoch.")
    parser.add_argument("--output_model_file", type=str, default="lstnet_model",
                        help="Name of the file to store the trained model.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate used in Adam optimizer.")
    parser.add_argument("--decay", type=float, nargs=2, default=(0.8, 0.999),
                        help="Two float values for Adam optimizer decay (beta1, beta2)")

    parser.add_argument("--delta_loss", type=float, default=1e-4,
                        help="Maximum allowed change in loss between iterations to consider convergence")


def add_translate_args(parser):
    parser.add_argument("domain", type=str.upper, help="Name of the domain to be translated to the other domain.")
    parser.add_argument("--load_model", action="store_true", help="If a model with name 'model_name' should be loaded for data translation.")
    parser.add_argument("--model_name", type=str, default="lstnet_model", help="Name of the model to be loaded for translation")
    parser.add_argument("--output_data_file", type=str, default="Name of the file to store the translated data.")


def add_eval_args(parser):
    parser.add_argument("domain", type=str.upper, help="Name of the domain to be evaluated.")
    parser.add_argument("clf_model", type=str, help="Name of the model to classify the data.")
    parser.add_argument("--output_results_file", type=str, help="Name of file to store test results")


def add_end_to_end_parser(parser):
    add_train_args(parser)

    parser.add_argument("clf_first_domain", type=str, help="Path to the trained classifier of the first domain")
    parser.add_argument("clf_second_domain", type=str, help="Path to the trained classifier of the second domain")
    parser.add_argument("--output_results_file", type=str, help="Name of file to store test results")
    parser.add_argument("--save_trans_data", action="store_true", help="Bool if the translated data that are result of the translation phase should be saved in files")
    parser.add_argument("--output_data_file", type=str, default="Name of the file to store the translated data. Only when '--save_trans_data' is set to true.")


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
    trans_parser = subparsers.add_parser("translate", help="Load a trained model and dataset and map the images from the original domain to the second one.")
    add_common_args(trans_parser)
    add_translate_args(trans_parser)

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Load a dataset and predict their labels using a given classifier")
    add_common_args(eval_parser)
    add_eval_args(eval_parser)

    # All: train->translate->evaluate
    all_parser = subparsers.add_parser("all", help="Perform the end-to-end workflow. Train the model. Load the test datasets and translate it to their respective target domains. Load a classifier and predict labels for the translated images.")
    add_common_args(all_parser)
    add_end_to_end_parser(all_parser)

    return parser.parse_args()


def initialize(args):
    utils.OUTPUT_FOLDER = args.output_folder

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    utils.NUM_WORKERS = args.num_workers
    utils.BATCH_SIZE = args.batch_size

    if args.operation in ['train', 'all']:
        utils.PARAMS_FILE_PATH = args.params_file
        utils.LOSS_FILE = args.loss_file
        utils.ADAM_LR = args.learning_rate
        utils.ADAM_DECAY = args.decay
        utils.DELTA_LOSS = args.delta_loss

    utils.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device being used: {utils.DEVICE}')


def run_training(first_domain, second_domain, supervised, output_file, return_model=False):
    model = train.run(args.first_domain, args.second_domain, args.supervised)

    model_path = f'{utils.OUTPUT_FOLDER}/{args.output_model_file}.pth'
    torch.save(model, model_path)

    if return_model:
        return model


def run_translation(args, domain_name, model=None, return_data=False):
    if model is None and args.load_model is False:
        raise ValueError("Model for translation is not specified.")

    if args.load_model:
        model = torch.load(args.model_name)

    translated_data = domain_adaptation.adapt_domain(model, domain_name)

    torch.save(translated_data, f'{utils.OUTPUT_FOLDER}/{args.output_data_file}.pt')

    if return_data:
        return translated_data


def run_evaluation(clf_name, domain_name, results_file):
    model = torch.load(clf_name, weights_only=False)
    test_acc = domain_adaptation.evaluate(model, domain_name)

    with open(f'{utils.OUTPUT_FOLDER}/{results_file}.json', 'a') as file:
        json.dump({f'{domain_name}_test_acc' : test_acc}, file, indent=2)


def run_end_to_end(args):
    model = run_training(args.first_domain, args.second_domain, args.supervised, args.output_model_file, return_model=True)

    first_data_trans = run_translation(args, args.first_domain, model, return_data=True)
    second_data_trans = run_translation(args, args.second_domain, model, return_data=True)

    if args.save_trans_data:
        torch.save(first_data_trans, f'{utils.OUTPUT_FOLDER}/{args.first_domain}_{args.output_data_file}.pt')
        torch.save(first_data_trans, f'{utils.OUTPUT_FOLDER}/{args.second_domain}_{args.output_data_file}.pt')

    first_clf = torch.load(args.clf_first_domain)
    run_evaluation(args, args.first_domain, first_clf)

    second_clf = torch.load(args.clf_second_domain)
    run_evaluation(args, args.second_domain, second_clf)


if __name__ == "__main__":
    args = parse_args()
    initialize(args)

    if args.operation == 'train':
        run_training(args.first_domain, args.second_domain, args.supervised, args.output_model_file)
        print(f'LSTNET model for {args.first_domain}-{args.second_domain} Domain Adaptation is trained.')

    elif args.operation == 'trans':
        run_translation(args, args.domain)

    elif args.operation == 'eval':
        run_evaluation(args.clf_model, args.domain, args.output_results_file)

    else:
        run_end_to_end(args)

