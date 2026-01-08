import torch
import argparse
import json
import os


import clf_utils
import utils
import clf_optuna
from data_preparation import AugmentOps, ResizeOps, ColorJitterOps


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
    _ = parser.add_argument("--strong_augment", action="store_true")
    
    # Strong augmentation parameters
    _ = parser.add_argument("--horizontal_flip_prob", type=float, default=0.3)
    _ = parser.add_argument("--color_jitter_brightness", type=float, default=0.3)
    _ = parser.add_argument("--color_jitter_contrast", type=float, default=0.3)
    _ = parser.add_argument("--color_jitter_saturation", type=float, default=0.3)
    _ = parser.add_argument("--color_jitter_hue", type=float, default=0.1)

    _ = parser.add_argument("--batch_size", type=int, default=64)
    _ = parser.add_argument("--num_workers", type=int, default=8)

    _ = parser.add_argument("--epoch_num", type=int, default=50)
    _ = parser.add_argument("--patience", type=int, default=5)
    _ = parser.add_argument("--optimizer", type=str, default="Adam")

    _ = parser.add_argument("--learning_rate", type=float, default=0.001)
    _ = parser.add_argument("--weight_decay", type=float, default=0.0)
    _ = parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))

    _ = parser.add_argument("--use_scheduler", action="store_true")
    _ = parser.add_argument("--scheduler_factor", type=float, default=0.1)
    _ = parser.add_argument("--scheduler_patience", type=int, default=5)
    _ = parser.add_argument("--scheduler_min_lr", type=float, default=1e-6)

    _ = parser.add_argument("--run_optuna", action="store_true")
    _ = parser.add_argument("--study_name", type=str, default="clf_study")
    _ = parser.add_argument("--n_trials", type=int, default=50)
    _ = parser.add_argument("--optuna_sampler_start_trials", type=int, default=20)
    _ = parser.add_argument("--min_resource", type=int, default=5)
    _ = parser.add_argument("--max_resource", type=int, default=20)
    _ = parser.add_argument("--reduction_factor", type=int, default=2)

    _ = parser.add_argument("--resize_target_size", type=int, default=None)
    _ = parser.add_argument("--pad_mode", type=str, default="edge")
    _ = parser.add_argument("--random_crop_resize", action="store_true")
    _ = parser.add_argument("--resize_init_size", type=int, default=256)
    _ = parser.add_argument("--resized_crop_scale_min", type=float, default=0.8)
    _ = parser.add_argument("--resized_crop_scale_max", type=float, default=1.0)
    _ = parser.add_argument("--resized_crop_ratio_min", type=float, default=0.9)
    _ = parser.add_argument("--resized_crop_ratio_max", type=float, default=1.1)
    _ = parser.add_argument("--inplace_augmentation", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.init_device()

    if args.run_optuna:
        print(f"Running Optuna for classifier on {args.domain_name}")
        trained_clf, logs = clf_optuna.run_optuna_clf(args)

    else:
        print(f"Training basic classifier on {args.domain_name}")
        
        # Create AugmentOps
        color_jitter = None
        if args.strong_augment:
            color_jitter = ColorJitterOps(
                brightness=args.color_jitter_brightness,
                contrast=args.color_jitter_contrast,
                saturation=args.color_jitter_saturation,
                hue=args.color_jitter_hue
            )
        
        augm_ops = None
        if args.aug_rotation != 0 or args.aug_zoom != 0.0 or args.aug_shift != 0 or args.strong_augment:
            augm_ops = AugmentOps(
                rotation=args.aug_rotation,
                zoom=args.aug_zoom,
                shift=args.aug_shift,
                use_strong_augment=args.strong_augment,
                horizontal_flip_prob=args.horizontal_flip_prob,
                color_jitter=color_jitter
            )
        
        # Create ResizeOps
        resize_ops = None
        if args.resize_target_size is not None:
            resize_ops = ResizeOps(
                target_size=args.resize_target_size,
                init_size=args.resize_init_size,
                pad_mode=args.pad_mode,
                random_crop_resize=args.random_crop_resize,
                resized_crop_scale=(args.resized_crop_scale_min, args.resized_crop_scale_max),
                resized_crop_ratio=(args.resized_crop_ratio_min, args.resized_crop_ratio_max),
            )
        
        train_loader, val_loader = clf_utils.prepare_clf_data(
            args.domain_name,
            val_size_data=args.val_size,
            seed=args.manual_seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_ops=augm_ops,
            resize_ops=resize_ops,
            inplace_augmentation=args.inplace_augmentation
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
            use_scheduler=args.use_scheduler,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            scheduler_min_lr=args.scheduler_min_lr,
        )

    if not os.path.exists(f"{args.output_folder}"):
        os.makedirs(f"{args.output_folder}")

    with open(f"{args.output_folder}/logs.json", "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=2)

    torch.save(trained_clf, f"{args.output_folder}/{args.domain_name}_clf_model.pth")
    print(
        f"Classifier model saved at: {args.output_folder}/{args.domain_name}_clf_model.pth"
    )
