import copy
import torch
import argparse
import json
import os
from time import time
import numpy as np

from data_preparation import load_dataset, create_augmentation_steps
from torch.utils.data import DataLoader, random_split
from eval_models.clf_models import MnistClf, UspsClf, SvhnClf

EVAL_FOLDER = 'eval_models/'
MODEL_FOLDER = None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('domain_name', type=str.upper)
    parser.add_argument('params_file', type=str)

    return parser.parse_args()


def run_loop(clf, loader, train=True):
    loss_total = 0
    acc_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        clf.optimizer.zero_grad()
        outputs = clf.forward(x)

        loss = clf.criterion(outputs, y)
        if train:
            loss.backward()
            clf.optimizer.step()

        loss_total += loss.item()  # reduction='sum' -> already returns sum of the losses

        preds = outputs.argmax(dim=1)
        acc = (preds == y).sum()
        acc_total += acc.item()

    loss_total /= len(loader.dataset)
    acc_total /= len(loader.dataset)

    return loss_total, acc_total


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # obtain the dataset size
    args.domain_name = args.domain_name.upper()
    MODEL_FOLDER = EVAL_FOLDER + args.domain_name

    if not os.path.exists(f'{MODEL_FOLDER}'):
        os.makedirs(f'{MODEL_FOLDER}')

    train_data = load_dataset(args.domain_name, train_op=True)
    img_size = train_data[0][0].shape[1]

    transform_steps = create_augmentation_steps(img_size)
    train_data = load_dataset(args.domain_name, train_op=True, download=False, transform_steps=transform_steps)

    train_size = int(len(train_data) * 0.75)
    val_size = len(train_data) - train_size

    train_data, val_data = random_split(train_data, [train_size, val_size])


    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4)

    clf = None
    if not args.params_file.endswith('.json'):
        raise ValueError("The parameter 'params_file' must end with .json")

    with open(f'{EVAL_FOLDER}{args.params_file}', 'r') as file:
        params = json.load(file)

    if args.domain_name == "MNIST":
        clf = MnistClf(params)
        print(f'MNIST Classifier Initialized')

    elif args.domain_name == "USPS":
        clf = UspsClf(params)
        print(f'USPS Classifier Initialized')

    elif args.domain_name == "SVHN":
        clf = SvhnClf(params)
        print(f'SVHN Classifier Initialized')

    if clf is None:
        raise ValueError("No classifier model as loaded.")
    clf.to(device)

    best_weights = copy.deepcopy(clf.state_dict())
    best_val_loss = np.inf

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    patience_cnt = 0

    for epoch in range(clf.epochs):
        start_time = time()
        ######################################################
        clf.train()
        train_loss, train_acc = run_loop(clf, train_loader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        ######################################################
        clf.eval()
        with torch.no_grad():
            val_loss, val_acc = run_loop(clf, val_loader, train=False)

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        ######################################################
        end_time = time()
        print(f'Epoch {epoch} finished.')
        print(f'\tTrain loss: {train_loss:.6f}, Train acc: {train_acc:.6f}')
        print(f'\tVal loss: {val_loss:.6f}, Val acc: {val_acc:.6f}')
        print(f'\tTook: {(end_time-start_time)/60:.2f} min')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(clf.state_dict())
            patience_cnt = 0

        else:
            patience_cnt += 1

            if patience_cnt > clf.patience:
                print(f'Patience {patience_cnt} reached its limit {clf.patience}.')
                break
        ######################################################

    results = {'train_loss': train_loss_list, 'train_acc': train_acc_list,
               'val_loss': val_loss_list, 'val_acc': val_acc_list}

    torch.save(best_weights, f"{MODEL_FOLDER}/best_weights.pth")

    with open(f'{MODEL_FOLDER}/results.json', 'w') as file:
        json.dump(results, file, indent=2)

    torch.save(clf.state_dict(), f"{MODEL_FOLDER}/{args.domain_name}_model.pth")
