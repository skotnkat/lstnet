import copy
import torch
import argparse
import json
import os
from time import time
import numpy as np

import clf_utils
from data_preparation import load_augmented_dataset
from torch.utils.data import DataLoader


import utils

EVAL_FOLDER = 'eval_models/'
MODEL_FOLDER = None
utils.MANUAL_SEED = 42
utils.VAL_SIZE = 0.25


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('domain_name', type=str.upper)
    parser.add_argument('params_file', type=str)

    return parser.parse_args()


def run_loop(clf, loader, train=True):
    loss_total = 0
    acc_total = 0
    for x, y in loader:
        x = x.to(utils.DEVICE)
        y = y.to(utils.DEVICE)
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

    loss_total /= len(loader)
    acc_total /= len(loader)

    return loss_total, acc_total


def load_data(domain_name):
    train_data, val_data = load_augmented_dataset(domain_name, split_data=True)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=8)

    return train_loader, val_loader


def train(domain_name, params, train_loader, val_loader, optuna=False, trial=None):
    clf = clf_utils.select_classifier(domain_name, params)

    best_clf = None
    best_val_acc = np.inf

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    patience_cnt = 0

    for epoch in range(clf.epochs):
        print(f'Epoch {epoch}:')
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
        print(f'\tTrain loss: {train_loss:.6f}, Train acc: {train_acc:.6f}')
        print(f'\tVal loss: {val_loss:.6f}, Val acc: {val_acc:.6f}')
        print(f'\tTook: {(end_time-start_time)/60:.2f} min')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_clf = copy.deepcopy(clf)
            patience_cnt = 0

        else:
            patience_cnt += 1

            if patience_cnt > clf.patience:
                print(f'Patience {patience_cnt} reached its limit {clf.patience}.')
                break

        print(f'patience: {patience_cnt}')

        if optuna:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        ######################################################
    results = {'train_loss': train_loss_list, 'train_acc': train_acc_list,
               'val_loss': val_loss_list, 'val_acc': val_acc_list}

    return best_clf, best_val_acc, results


if __name__ == "__main__":
    args = parse_args()
    utils.assign_device()

    # obtain the dataset size
    args.domain_name = args.domain_name.upper()
    MODEL_FOLDER = EVAL_FOLDER + args.domain_name

    if not os.path.exists(f'{MODEL_FOLDER}'):
        os.makedirs(f'{MODEL_FOLDER}')

    if not args.params_file.endswith('.json'):
        raise ValueError("The parameter 'params_file' must end with .json")

    with open(f'{EVAL_FOLDER}{args.params_file}', 'r') as file:
        params = json.load(file)

    train_loader, val_loader = load_data(args.domain_name)

    clf, val_acc, results = train(args.domain_name, params, train_loader, val_loader)

    with open(f'{MODEL_FOLDER}/results.json', 'w') as file:
        json.dump(results, file, indent=2)

    print(f'Best validation loss: {val_acc}')

    torch.save(clf, f"{MODEL_FOLDER}/{args.domain_name}_model.pth")

