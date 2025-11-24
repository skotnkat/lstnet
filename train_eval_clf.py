import torch
import argparse
import json
import os


import clf_utils
import utils
import clf_optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    _ = parser.add_argument("domain_name", type=str.upper)
    _ = parser.add_argument(
        "--params_file", type=str
    )  # make the changes, so it is only optional -> not needed for optuna
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

    _ = parser.add_argument("--run_optuna", action="store_true")
    _ = parser.add_argument("--study_name", type=str, default="clf_study")
    _ = parser.add_argument("--n_trials", type=int, default=50)
    _ = parser.add_argument("--optuna_sampler_start_trials", type=int, default=20)
    _ = parser.add_argument("--min_resource", type=int, default=5)
    _ = parser.add_argument("--max_resource", type=int, default=20)
    _ = parser.add_argument("--reduction_factor", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_device()

    if args.run_optuna:
        print(f"Running Optuna for classifier on {args.domain_name}")
        trained_clf, logs = clf_optuna.run_optuna_clf(args)

    else:
        print(f"Training basic classifier on {args.domain_name}")
        train_loader, val_loader = clf_utils.prepare_clf_data(
            args.domain_name,
            val_size_data=args.val_size,
            seed=args.manual_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rotation=args.aug_rotation,
            zoom=args.aug_zoom,
            shift=args.aug_shift,
        )
        print("Data preppared.")

        clf = clf_utils.get_clf(args.domain_name, params_path=args.params_file)
        print("Clf obtained.")

        print("Starting training...")

        trained_clf, _, logs = clf_utils.train_clf(
            clf,
            train_loader,
            val_loader,
            optim=args.optimizer,
            epoch_num=args.epoch_num,
            patience=args.patience,
            lr=args.learning_rate,
            betas=args.betas,
            weight_decay=args.weight_decay,
        )

    if not os.path.exists(f"{args.output_folder}"):
        os.makedirs(f"{args.output_folder}")

    with open(f"{args.output_folder}/logs.json", "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=2)

    torch.save(trained_clf, f"{args.output_folder}/{args.domain_name}_clf_model.pth")
    print(
        f"Classifier model saved at: {args.output_folder}/{args.domain_name}_clf_model.pth"
    )
