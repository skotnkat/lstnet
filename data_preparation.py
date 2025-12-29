"""
Module for preparing datasets for training, evaluation and testing for the LSTNET model.
"""

from typing import (
    Literal,
    Union,
    Tuple,
    Any,
    overload,
    TypeAlias,
    Optional,
    Callable,
    List,
)
from dataclasses import dataclass
import os
from PIL import Image
from collections import defaultdict

from torchvision import datasets
from torchvision.transforms.v2 import (
    Compose,
    RandomAffine,
    ToImage,
    ToDtype,
    Normalize,
    Lambda, 
    RandomCrop
)
from torch.utils.data import ConcatDataset, DataLoader, random_split, Dataset, Subset
import torch
from dual_domain_dataset import get_dual_domain_dataset, custom_collate_fn
import download_data
import random

import torchvision.transforms.functional as F


SingleLoader: TypeAlias = DataLoader[Any]
DoubleLoader: TypeAlias = Tuple[DataLoader[Any], DataLoader[Any]]

SingleDataset: TypeAlias = Dataset[Any]
DoubleDataset: TypeAlias = Tuple[Dataset[Any], Dataset[Any]]


@dataclass(slots=True)
class AugmentOps:
    """Data class for augmentation operations. Holds parameters for augmentation.
    Args:
        rotation (int): The maximum rotation angle in degrees. Defaults to 10.
        zoom (float): The zoom factor. Defaults to 0.1.
        shift (int): The maximum shift in pixels. Defaults to 2.
    """

    rotation: int = 10
    zoom: float = 0.1
    shift: int = 2


@dataclass(slots=True)
class ResizeOps:
    target_size: int = 224
    init_size: int = 256
    pad_mode: str = 'edge'
    random_crop_resize: bool = False
    
    def __post_init__(self):
        if self.target_size > self.init_size:
            raise ValueError(f"target_size ({self.target_size}) should be less than or equal to init_size ({self.init_size})")
        

def get_augmentation_steps(
    img_size: Tuple[int, int],
    *,
    augment_ops: Optional[AugmentOps] = None,
    fill: Union[int, Tuple[int, ...]] = 0,
) -> List[Any]:  # TODO: specify correct type
    """
    Creates a series of augmentation steps for image preprocessing.
    Expecting to work with square 2D images.

    Args:
        img_size (int): The size of the input images.
        num_channels (int, optional): The number of channels in the input images. Defaults to 1.
        augment_ops (AugmentOps, optional): The augmentation operations to apply.
            Defaults to AugmentOps().

    Returns:
        Compose: A composition of image transformation steps.
    """
    print(f"augment_ops: {augment_ops}")
    dx_translation = augment_ops.shift / img_size[0]
    dy_translation = augment_ops.shift / img_size[1]

    return [
        RandomAffine(
            degrees=(-augment_ops.rotation, augment_ops.rotation),
            translate=(dx_translation, dy_translation),
            scale=(1 - augment_ops.zoom, 1 + augment_ops.zoom),
            fill=fill
        )
    ]


def resize_with_padding(
    img: torch.Tensor,
    target_size: int,
    pad_mode: str = 'edge',
) -> torch.Tensor:
    """
    Resize image to fit within target_size while preserving aspect ratio,
    then pad to create a square image.
    
    Args:
        img: Input image tensor (C, H, W)
        target_size: Target square size (e.g., 256 or 224)
        pad_mode: Padding mode - 'edge', 'reflect', 'symmetric', or 'constant'
    
    Returns:
        Square image tensor (C, target_size, target_size)
    """
    
    _, height, width = img.shape
    
    # Calculate scale to fit within target_size
    scale = target_size / max(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    
    # Resize maintaining aspect ratio
    img = F.resize(img, [new_height, new_width], antialias=True)
    
    # Calculate padding needed to fill into square
    pad_h = target_size - new_height
    pad_w = target_size - new_width
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    
    return F.pad(img, padding, padding_mode=pad_mode)


def create_transform_steps(
    num_channels: int = 1,
    *,
    img_size: Optional[Tuple[int, int]] = None,
    augment_ops: Optional[AugmentOps] = None,
    resize_ops: Optional[ResizeOps] = None,
) -> Compose:
    """
    Create a basic transform (type+normalization) for a given channel count.
    Args:
        num_channels (int, optional): The number of channels in the input images. Defaults to 1.
    """
    # TODO: update docstring

    ops: List[Any] = [ToImage()]

    if resize_ops is not None:
        first_size = resize_ops.target_size
        if resize_ops.random_crop_resize is True:
            first_size = resize_ops.init_size
        # Resize images to the specified size (max size) and pad the smaller dimension with 0 to create square (resize, resize)
        ops.append(
            Lambda(lambda img: resize_with_padding(img, first_size, pad_mode=resize_ops.pad_mode))
            )
        
        if resize_ops.random_crop_resize is True:
            ops.append(
                RandomCrop(resize_ops.target_size)
            )
            
            img_size = (resize_ops.target_size, resize_ops.target_size)

    if augment_ops is not None:
        if img_size is None:
            raise ValueError("img_size must be provided when augment_ops is used")

        fill = 0
        if num_channels == 3:
            fill = (127, 127, 127)
        ops.extend(get_augmentation_steps(img_size, augment_ops=augment_ops, fill=fill))

    ops.extend(
        [
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
        ]
    )

    return Compose(ops)


class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        transform=None,
        extensions: List[str] = [".jpg"],
        rgb: bool = True,
        dummy_class: int = 1,
    ):
        """
        Args:
            folder: Folder path to load images from
            transform: Optional torchvision transforms to apply
        """
        self.image_paths = []
        self.transform = transform
        self.rgb = rgb
        self.extensions = extensions
        self.dummy_class = dummy_class

        # Collect all image paths from the folder
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
        else:
            for filename in os.listdir(folder):
                if any(filename.lower().endswith(ext) for ext in self.extensions):
                    self.image_paths.append(os.path.join(folder, filename))

        print(f"Found {len(self.image_paths)} images in {folder}")

        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = Image.open(img_path)

        if self.rgb:
            image = image.convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, self.dummy_class


def get_a2o_dataset(
    dataset: str,
    *,
    train_op: bool,
    transform_steps: Optional[Compose] = None,
    domain_adaptation: bool = False,
) -> Dataset[Any]:
    target_path = "data/a2o_dataset"
    download_data.download_a2o_dataset(target_path)

    if transform_steps is None:
        transform_steps = create_transform_steps(3)

    dummy_class = 1
    folder = "test"
    if train_op:
        folder = "train"

    letter = "A"
    if dataset.upper() == "ORANGE":
        letter = "B"
        dummy_class = 0

    # Switch Labels
    if domain_adaptation:
        dummy_class = 1 - dummy_class

    path = f"{target_path}/{folder}{letter}"
    print(f"path: {path}")
    data = ImageDataset(
        folder=path,
        transform=transform_steps,
        dummy_class=dummy_class,
    )

    return data


def get_a2o_clf_dataset(
    *,
    train_op: bool,
    transform_steps: Optional[Compose] = None,
) -> Dataset[Any]:
    apple_data = get_a2o_dataset(
        "APPLE", train_op=train_op, transform_steps=transform_steps
    )
    orange_data = get_a2o_dataset(
        "ORANGE", train_op=train_op, transform_steps=transform_steps
    )

    return ConcatDataset([apple_data, orange_data])


def get_data_loader(
    data: Dataset[Any],
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 8,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Creates a DataLoader for the given dataset.

    Args:
        data (Dataset[Any]): The dataset passed to the loader.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): The number of subprocesses to use for data loading.
            Defaults to 8.
        persistent_workers (bool, optional): Whether to keep workers alive after the dataset
            has been consumed. Defaults to False.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer to the GPU.
            Defaults to False.
        collate_fn (Optional[Callable], optional): The function to merge samples into batches.
            Defaults to None. If None, the default collate function is used.

    Returns:
        DataLoader[Any]: The DataLoader instance for the given dataset.
    """

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


# Overloads for type checking
@overload
def load_dataset(
    dataset_name: str,
    *,
    split_data: Literal[True],
    op: str = "train",
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,
    resize_ops: Optional[ResizeOps] = None,
) -> DoubleDataset: ...
@overload
def load_dataset(
    dataset_name: str,
    *,
    split_data: Literal[False],
    op: str = "train",
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,
    resize: Optional[int] = None,
    resize_ops: Optional[ResizeOps] = None,
) -> SingleDataset: ...


# Single runtime implementation
def load_dataset(
    dataset_name: str,
    *,
    split_data: bool = False,
    op: str = "train",
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,  # switch labels for A2O
    resize_ops: Optional[ResizeOps] = None,
) -> Union[SingleDataset, DoubleDataset]:
    """Loads a dataset from torchvision or a local path.

    Args:
        dataset_name (str): The name of the dataset to load.
        split_data (bool, optional): Whether to split the data into training and validation sets.
            Defaults to False.
        train_op (bool, optional): Whether to load the training or test split of the dataset.
            Defaults to training split.
        transform_steps (Compose, optional): The transformation steps to apply to the dataset.
            If None, a basic transformation is applied (conversion to float and normalization).
        download (bool, optional): In case of torchvision dataset,
            whether to download the dataset if not found locally.
            Defaults to True.
        val_data_size (float, optional): The proportion of the dataset to use for validation.
            Applied only when split_data is True. Defaults to 0.4.
        manual_seed (int, optional): The seed for random number generation.
            Defaults to 42.
        resize_ops: Optional[ResizeOps] = None,
            Resize operations to apply during data loading.
    Returns:
        Union[Dataset[Any], Tuple[Dataset[Any], Dataset[Any]]]: The loaded dataset(s).
            Either whole training dataset or a tuple of (training, validation) datasets.
    """
    train_op = op.lower() == "train"
    # load data from torchvision datasets
    data: Dataset[Any]
    match dataset_name.upper():
        case "MNIST":
            if transform_steps is None:
                transform_steps = create_transform_steps(1, resize_ops=resize_ops)

            data = datasets.MNIST(
                root="./data",
                train=train_op,
                transform=transform_steps,
                download=download,
            )

        case "USPS":
            if transform_steps is None:
                transform_steps = create_transform_steps(1, resize_ops=resize_ops)

            data = datasets.USPS(
                root="./data",
                train=train_op,
                transform=transform_steps,
                download=download,
            )

        case "SVHN":
            if transform_steps is None:
                transform_steps = create_transform_steps(3, resize_ops=resize_ops)

            if op.startswith("extra"):
                train_op = True if op == "extra_train" else False
                op = "extra"
            data = datasets.SVHN(
                root="./data", split=op, transform=transform_steps, download=download
            )
            
            if op == "extra":
                train_data, test_data = split_train_val_dataset(
                    data, val_data_size=0.2, manual_seed=manual_seed
                )  # splitting to train and test

                if train_op:
                    data = train_data

                else:
                    data = test_data


        case "APPLE":  # from the a2o dataset
            data = get_a2o_dataset(
                "APPLE",
                train_op=train_op,
                transform_steps=transform_steps,
                domain_adaptation=domain_adaptation,
            )

        case "ORANGE":  # from the a2o dataset
            data = get_a2o_dataset(
                "ORANGE",
                train_op=train_op,
                transform_steps=transform_steps,
                domain_adaptation=domain_adaptation,
            )

        case "A2O":  # Combined Apple and Orange for classification
            data = get_a2o_clf_dataset(
                train_op=train_op, transform_steps=transform_steps
            )

        case name if name.startswith("VISDA"):
            if download:
                download_data.download_visda_dataset(train_op=train_op)

            subfolder = "train"  #  VISDA_SOURCE
            if "VISDA_TARGET":
                subfolder = "validation"

            elif "VISDA_TEST":
                subfolder = "test"

            data_folder = download_data.DATA_FOLDER + "/" + subfolder

            if transform_steps is None:
                transform_steps = create_transform_steps(3, resize_ops=resize_ops)

            data = datasets.ImageFolder(
                data_folder,
                transform=transform_steps,
            )

        case name if name.startswith("OFFICE_31"):
            target_path = "data/office_31"
            if download:
                download_data.download_office_31_dataset(target_path)

            subfolder = "amazon/images"

            if name == "OFFICE_31_WEBCAM":
                subfolder = "webcam/images"

            elif name == "OFFICE_31_DSLR":
                subfolder = "dslr/images"

            if transform_steps is None:
                transform_steps = create_transform_steps(3, resize_ops=resize_ops)

            data_folder = f"{target_path}/{subfolder}"
            data = datasets.ImageFolder(data_folder, transform=transform_steps)

            train_data, test_data = split_train_val_dataset(
                data, val_data_size=0.25, manual_seed=manual_seed
            )  # splitting to train and test

            data = test_data
            if train_op:
                data = train_data

        case name if name.startswith("HOME_OFFICE"):
            target_path = "data/home_office/OfficeHomeDataset_10072016"
            if download:
                download_data.download_home_office_dataset(target_path)

            subfolder = "Art"
            if name == "HOME_OFFICE_CLIPART":
                subfolder = "Clipart"
            elif name == "HOME_OFFICE_REALWORLD":
                subfolder = "Real World"

            if transform_steps is None:
                transform_steps = create_transform_steps(3, resize_ops=resize_ops)  # TODO: add resize
                
            # TODO: refactor to not do the same code twice
            data_folder = f"{target_path}/{subfolder}"
            data = datasets.ImageFolder(data_folder, transform=transform_steps)

            train_data, test_data = split_train_val_dataset(
                data, val_data_size=0.25, manual_seed=manual_seed
            )  # splitting to train and test

            data = test_data
            if train_op:
                data = train_data

        case _:
            # dataset_name is path
            print(f"Trying to load dataset {dataset_name} locally")
            data = torch.load(dataset_name)  # type: ignore
            print(
                f"Dataset loaded, number of records: {len(data)}, shape: {data[0][0].shape}"  # type: ignore  (has len())
            )

    if split_data:
        return split_train_val_dataset(
            data, val_data_size=val_data_size, manual_seed=manual_seed
        )

    return data


def split_train_val_dataset(
    data: Dataset[Any], *, val_data_size: float = 0.4, manual_seed: int = 42
) -> DoubleDataset:
    """
    Splits the dataset into training and validation sets.

    Args:
        data (Dataset[Any]): The dataset to split.
        val_data_size (float, optional): The proportion of the dataset to use for validation.
            Defaults to 0.4.
        manual_seed (int, optional): The seed for random number generation. Defaults to 42.

    Returns:
        Tuple[Dataset[Any], Dataset[Any]]: The training and validation datasets.

    """

    g = torch.Generator().manual_seed(manual_seed)

    val_size = int(len(data) * val_data_size)  # type: ignore  (has len())
    train_size = len(data) - val_size  # type: ignore  (has len())

    train_data: Dataset[Any]
    val_data: Dataset[Any]
    train_data, val_data = random_split(data, [train_size, val_size], generator=g)

    return train_data, val_data


def get_dataset_chw(
    dataset: Dataset[Any], *, square_expected: bool = True
) -> Tuple[int, int, int]:
    """
    Finds the dimensions (C, H, W) of a single dataset from its first sample.
    Args:
        dataset (Dataset[Any]): The dataset to get dimensions from.
        square_expected (bool, optional): Whether to expect square images. Defaults to True.

    Raises:
        ValueError: If square images are expected but the first sample is not square.

    Returns:
        Tuple[int, int, int]: The dimensions (C, H, W) of the dataset.
    """
    sample = dataset[0][0]
    print(f"shape: {sample.shape}")
    if square_expected and (sample.shape[1] != sample.shape[2]):
        raise ValueError("Expected square images")

    return int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2])


def get_balanced_subset(
    dataset: Dataset[Any],
    *,
    manual_seed: int = 42,
) -> Dataset[Any]:
    """
    Create a balanced subset from a dataset with equal number of samples per class.
    
    Args:
        dataset (Dataset[Any]): The source dataset to sample from.
        samples_per_class (int): Number of samples to select from each class.
        manual_seed (int, optional): Seed for reproducibility. Defaults to 42.
    
    Returns:
        Dataset[Any]: A Subset with balanced class distribution.
        
    """

    
    random.seed(manual_seed)
    
    # Organize indices by class label
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):  # type: ignore
        _, label = dataset[idx]
        class_indices[label].append(idx)
        
    samples_per_class = min(len(indices) for indices in class_indices.values())
    print(f"Balancing to smallest class size: {samples_per_class} samples per class")
    
    # Sample equally from each class
    selected_indices = []
    for label, indices in sorted(class_indices.items()):
        if len(indices) < samples_per_class:
            raise ValueError(
                f"Class {label} has only {len(indices)} samples, "
                f"but {samples_per_class} requested"
            )
        sampled = random.sample(indices, samples_per_class)
        selected_indices.extend(sampled)
    
    # Shuffle the combined indices
    random.shuffle(selected_indices)
    
    print(f"Created balanced subset with {len(selected_indices)} samples "
          f"({samples_per_class} per class, {len(class_indices)} classes)")
    
    return Subset(dataset, selected_indices)

@overload
def load_augmented_dataset(
    dataset_name: str,
    *,
    split_data: Literal[False],
    train_op: bool = True,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    use_svhn_extra: bool = False,
) -> SingleDataset: ...
@overload
def load_augmented_dataset(
    dataset_name: str,
    *,
    split_data: Literal[True],
    train_op: bool = True,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    use_svhn_extra: bool = False,
) -> DoubleDataset: ...


def load_augmented_dataset(
    dataset_name: str,
    *,
    split_data: bool = False,
    train_op: bool = True,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    use_svhn_extra: bool = False,
) -> Union[SingleDataset, DoubleDataset]:
    """
    Augmented dataset by combining original dataset and augmented dataset.
    Loads dataset, creates augmentation steps and loads the dataset again with augmentations.

    Args:
        dataset_name (str): The name of the dataset to load.
        train_op (bool, optional): Whether to load the training set. Defaults to True.
        download (bool, optional): Whether to download the dataset if not found. Defaults to True.
        split_data (bool, optional): Whether to split the data into training and validation sets.
            Defaults to False.
        val_data_size (float, optional): The proportion of the dataset to use for validation.
            Applied only when split_data is True. Defaults to 0.4.
        manual_seed (int, optional): The seed for random number generation.
            Defaults to 42.
        augment_ops (AugmentOps, optional): The augmentation operations to apply.
    Returns:
        Union[Dataset[Any], Tuple[Dataset[Any], Dataset[Any]]]: The loaded augmented dataset(s).
    """
    print(f"Loading dataset: {dataset_name}")
    op = "train" if train_op else "test"
    
    if use_svhn_extra:
        op = "extra_" + op 
        
        if dataset_name.upper() != "SVHN":
            raise ValueError("use_svhn_extra can only be True for SVHN dataset")
        
        
    
    original_data = load_dataset(
        dataset_name,
        op=op,
        split_data=split_data,
        download=download,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        resize=resize,
    )

    ref_ds = original_data
    if split_data:
        ref_ds = original_data[0]

    # img height and width are same
    num_channels, img_size, _ = get_dataset_chw(
        ref_ds, square_expected=True  # type: ignore
    )

    transform_steps = create_transform_steps(
        num_channels,
        img_size=(img_size, img_size),
        augment_ops=augment_ops,
        resize=resize,
    )

    augmented_data = load_dataset(
        dataset_name,
        op=op,
        download=False,
        split_data=split_data,
        transform_steps=transform_steps,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
    )
    
    
    
    if skip_augmentation:  # more like inplace augmentation
        if use_svhn_extra:
            #TODO: fix if not split
            train_augmented, val_augmented = augmented_data
            train_augmented_balanced = get_balanced_subset(train_augmented, manual_seed=manual_seed)
            print(f"After balancing, training set size: {len(train_augmented_balanced)}")
            return train_augmented_balanced, val_augmented
        


    if not split_data:
        assert not isinstance(original_data, tuple)
        assert not isinstance(augmented_data, tuple)
        
        data = ConcatDataset([original_data, augmented_data])
            
        return data

    orig_train, orig_val = original_data
    augm_train, _ = augmented_data

    train_data: Dataset[Any] = ConcatDataset([orig_train, augm_train])
    

    return train_data, orig_val

#TODO: remove drop last from the logic
def get_train_val_loaders(
    first_domain_name: str,
    second_domain_name: str,
    supervised: bool,
    *,
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
    use_svhn_extra: bool = False,
) -> DoubleLoader:
    """
    Get training and validation data loaders for dual domain datasets.

    Args:
        first_domain_name (str): The name of the first domain dataset.
        second_domain_name (str): The name of the second domain dataset.
        supervised (bool): Whether the task is supervised or unsupervised.
        manual_seed (int, optional): The seed for random number generation.
            Defaults to 42.
        val_data_size (float, optional): The proportion of the dataset to use for validation.
            Defaults to 0.4.
        augment_ops (AugmentOps, optional): The augmentation operations to apply.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer to the GPU.
            Defaults to False.
    Returns:
        Tuple[DataLoader[Any], DataLoader[Any]]: The training and validation data loaders.
    """

    print("Skipping first domain augmnetation")
    first_train, first_val = load_augmented_dataset(
        first_domain_name,
        train_op=True,
        split_data=True,
        augment_ops=augment_ops,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        skip_augmentation=True,
        resize=resize,
        use_svhn_extra=use_svhn_extra if first_domain_name.upper() == "SVHN" else False,
    )
    second_train, second_val = load_augmented_dataset(
        second_domain_name,
        train_op=True,
        split_data=True,
        augment_ops=augment_ops,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        skip_augmentation=skip_augmentation,
        resize=resize,
        use_svhn_extra=use_svhn_extra if second_domain_name.upper() == "SVHN" else False,
    )

    val_data = get_dual_domain_dataset(first_val, second_val, supervised)
    train_data = get_dual_domain_dataset(first_train, second_train, supervised)

    train_loader = get_data_loader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False
    )
    val_loader = get_data_loader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False
    )

    print("Obtained Data Loader for both training and validation")

    return train_loader, val_loader


@overload
def get_training_loader(
    first_domain_name: str,
    second_domain_name: str,
    supervised: bool = True,
    *,
    split_data: Literal[False],
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    use_svhn_extra: bool = False,
) -> SingleLoader: ...
@overload
def get_training_loader(
    first_domain_name: str,
    second_domain_name: str,
    supervised: bool = True,
    *,
    split_data: Literal[True],
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    augment_ops: Optional[AugmentOps] = None,
    use_svhn_extra: bool = False,
) -> DoubleLoader: ...


def get_training_loader(
    first_domain_name: str,
    second_domain_name: str,
    supervised: bool = True,
    *,
    split_data: bool,
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
    augment_ops: AugmentOps = AugmentOps(),
    skip_augmentation: bool = False,
    resize: Optional[int] = None,
    use_svhn_extra: bool = False,
) -> Union[SingleLoader, DoubleLoader]:
    """
    Get the data loaders for a dual domain dataset for training phase.
    In case the data is not split, the returned loader contains the whole training data.
    Otherwise, two separate loaders are returned (training and validation).

    Args:
        first_domain_name (str): The name of the first domain dataset.
        second_domain_name (str): The name of the second domain dataset.
        supervised (bool, optional): Whether the task is supervised or unsupervised.
            Defaults to True.
        split_data (bool, optional): Whether to split the data into training and validation sets.
            Defaults to False.
        manual_seed (int, optional): The seed for random number generation.
            Defaults to 42.
        val_data_size (float, optional): The proportion of the dataset to use for validation.
            Defaults to 0.4.
        augment_ops (AugmentOps, optional): The augmentation operations to apply.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
        pin_memory (bool, optional): Whether to pin memory for faster data transfer to the GPU.
            Defaults to False.

    Returns:
        DataLoader: The training data loader(s).
    """

    if split_data:
        return get_train_val_loaders(
            first_domain_name,
            second_domain_name,
            supervised,
            manual_seed=manual_seed,
            val_data_size=val_data_size,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_ops=augment_ops,
            skip_augmentation=skip_augmentation,
            resize=resize,
            pin_memory=pin_memory,
            use_svhn_extra=use_svhn_extra,
        )

    print(f'Skippin augmentation for first domain: {first_domain_name}')
    first_data = load_augmented_dataset(
        first_domain_name,
        train_op=True,
        split_data=False,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        augment_ops=augment_ops,
        skip_augmentation=True,
        resize=resize,
        use_svhn_extra=use_svhn_extra if first_domain_name.upper() == "SVHN" else False,
    )
    second_data = load_augmented_dataset(
        second_domain_name,
        train_op=True,
        split_data=False,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        augment_ops=augment_ops,
        skip_augmentation=skip_augmentation,
        resize=resize,
        use_svhn_extra=use_svhn_extra if second_domain_name.upper() == "SVHN" else False,
    )

    dual_data = get_dual_domain_dataset(first_data, second_data, supervised)

    # Respect the provided pin_memory value; set True at call sites if training on GPU.

    data_loader = get_data_loader(
        dual_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        collate_fn=custom_collate_fn,
        drop_last=True,  # To ensure consistent batch sizes during training
    )
    print("Obtained Data Loader for training")

    return data_loader


def get_testing_loader(
    domain_name: str,
    *,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
    domain_adaptation: bool = False,
) -> DataLoader[Any]:
    """Get the data loader of test set for the given dataset (domain name).

    Args:
        domain_name (str): The name of the domain dataset.
        batch_size (int, optional): The batch size for the data loader. Defaults to 32.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.

    Returns:
        DataLoader[Any]: The data loader for testing phase.
    """

    data = load_dataset(
        domain_name,
        op="test",
        split_data=False,
        domain_adaptation=domain_adaptation,
    )

    # For testing, no need to shuffle the data
    return get_data_loader(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
