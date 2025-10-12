import torch
import argparse
import json
import os

from eval_models.clf_models import select_classifier, ClfTrainer
from data_preparation import load_augmented_dataset, AugmentOps
from torch.utils.data import DataLoader


import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("domain_name", type=str.upper)
    _ = parser.add_argument("params_file", type=str)
    _ = parser.add_argument("--output_folder", type=str, default="eval_models/")
    # _ = parser.add_argument("--custom_clf", action="store_true")

    _ = parser.add_argument("--manual_seed", type=int, default=42)
    _ = parser.add_argument("--val_size", type=float, default=0.25)

    _ = parser.add_argument("--aug_rotation", type=int, default=10)
    _ = parser.add_argument("--aug_zoom", type=float, default=0.1)
    _ = parser.add_argument("--aug_shift", type=int, default=2)

    _ = parser.add_argument("--batch_size", type=int, default=64)
    _ = parser.add_argument("--num_workers", type=int, default=8)

    _ = parser.add_argument("--epoch_num", type=int, default=50)
    _ = parser.add_argument("--patience", type=int, default=5)
    _ = parser.add_argument("--optimizer", type=str, default="Adam")

    _ = parser.add_argument("--learning_rate", type=float, default=0.001)
    _ = parser.add_argument("--weight_decay", type=float, default=0.0)
    _ = parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_device()

    # Load Trainining and Validation Data
    aug_ops = AugmentOps(
        rotation=args.aug_rotation, zoom=args.aug_zoom, shift=args.aug_shift
    )

    ds_name: str = args.domain_name.upper()
    val_size: float = args.val_size
    manual_seed: int = args.manual_seed
    train_data, val_data = load_augmented_dataset(
        ds_name,
        split_data=True,
        train_op=True,
        val_data_size=val_size,
        manual_seed=manual_seed,
        augment_ops=aug_ops,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load Parameters File
    with open(f"{args.params_file}", "r", encoding="utf-8") as file:
        params = json.load(file)

    clf = select_classifier(args.domain_name.upper(), params=params)
    trainer = ClfTrainer(
        clf,
        optimizer=args.optimizer,
        epochs=args.epoch_num,
        patience=args.patience,
        lr=args.learning_rate,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
    )

    clf = trainer.train(clf, train_loader, val_loader)

    if not os.path.exists(f"{args.output_folder}"):
        os.makedirs(f"{args.output_folder}")

    trainer_info = trainer.get_trainer_info()

    with open(f"{args.output_folder}/logs.json", "w", encoding="utf-8") as file:
        json.dump(trainer_info, file, indent=2)

    print(f"Best validation accuracy: {trainer.best_acc}")

    torch.save(clf, f"{args.output_folder}/{args.domain_name}_clf_model.pth")
    print(
        f"Classifier model saved at: {args.output_folder}/{args.domain_name}_clf_model.pth"
    )
