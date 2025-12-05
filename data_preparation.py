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
import tarfile
import urllib.request
import kagglehub
from PIL import Image

from torchvision import datasets
from torchvision.transforms.v2 import Compose, RandomAffine, ToImage, ToDtype, Normalize
from torch.utils.data import ConcatDataset, DataLoader, random_split, Dataset
import torch
from dual_domain_dataset import get_dual_domain_dataset, custom_collate_fn


SingleLoader: TypeAlias = DataLoader[Any]
DoubleLoader: TypeAlias = Tuple[DataLoader[Any], DataLoader[Any]]

SingleDataset: TypeAlias = Dataset[Any]
DoubleDataset: TypeAlias = Tuple[Dataset[Any], Dataset[Any]]


A2O_DATASET = "balraj98/apple2orange-dataset"


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


def create_augmentation_steps(
    img_size: int, *, num_channels: int = 1, augment_ops: AugmentOps = AugmentOps()
) -> Compose:
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
    dx_translation = dy_translation = augment_ops.shift / img_size

    return Compose(
        [
            ToImage(),
            RandomAffine(
                degrees=(-augment_ops.rotation, augment_ops.rotation),
                translate=(dx_translation, dy_translation),
                scale=(1 - augment_ops.zoom, 1 + augment_ops.zoom),
            ),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
        ]
    )


def create_basic_transform(num_channels: int = 1) -> Compose:
    """
    Create a basic transform (type+normalization) for a given channel count.
    Args:
        num_channels (int, optional): The number of channels in the input images. Defaults to 1.
    """
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
        ]
    )


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
    cache_path = kagglehub.dataset_download(A2O_DATASET)

    if transform_steps is None:
        transform_steps = create_basic_transform(3)

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

    path = f"{cache_path}/{folder}{letter}"
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
    train_op: bool = True,
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,
) -> DoubleDataset: ...
@overload
def load_dataset(
    dataset_name: str,
    *,
    split_data: Literal[False],
    train_op: bool = True,
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,
) -> SingleDataset: ...


# Single runtime implementation
def load_dataset(
    dataset_name: str,
    *,
    split_data: bool = False,
    train_op: bool = True,
    transform_steps: Optional[Compose] = None,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    domain_adaptation: bool = False,  # switch labels for A2O
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

    Returns:
        Union[Dataset[Any], Tuple[Dataset[Any], Dataset[Any]]]: The loaded dataset(s).
            Either whole training dataset or a tuple of (training, validation) datasets.
    """

    # load data from torchvision datasets
    data: Dataset[Any]
    match dataset_name.upper():
        case "MNIST":
            if transform_steps is None:
                transform_steps = create_basic_transform(1)

            data = datasets.MNIST(
                root="./data",
                train=train_op,
                transform=transform_steps,
                download=download,
            )

        case "USPS":
            if transform_steps is None:
                transform_steps = create_basic_transform(1)

            data = datasets.USPS(
                root="./data",
                train=train_op,
                transform=transform_steps,
                download=download,
            )

        case "SVHN":
            # Different way of getting train/test sets
            split = "test"
            if train_op:
                split = "train"

            if transform_steps is None:
                transform_steps = create_basic_transform(3)

            data = datasets.SVHN(
                root="./data", split=split, transform=transform_steps, download=download
            )

            if transform_steps is None:
                transform_steps = create_basic_transform(3)

            data = datasets.SVHN(
                root="./data", split=split, transform=transform_steps, download=download
            )

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


# def load_benchmark(
#     benchmark_name: str,
#     *,
#     split_data: bool = False,
#     train_op: bool = True,
#     transform_steps: Optional[Compose] = None,
#     val_data_size: float = 0.4,
#     manual_seed: int = 42):

#     match benchmark_name:
#         case "Visda2017":
#             BASE_PATH = "http://csr.bu.edu/ftp/visda17/clf"

#             FOLDER = "data/visda2017"
#             os.makedirs(FOLDER, exist_ok=True)

#             # Check if files are already downloaded, if not -> download
#             if train_op:
#                 if not os.path.exists(FOLDER + "/source.tar"):
#                     _ = urllib.request.urlretrieve(
#                         BASE_PATH + "/train.tar", FOLDER + "/source.tar"
#                     )

#                 if not os.path.exists(FOLDER + "/target.tar"):
#                     _ = urllib.request.urlretrieve(
#                         BASE_PATH + "/validation.tar", FOLDER + "/target.tar"
#                     )

#                 with tarfile.open(FOLDER + "/source.tar", "r") as tar:
#                     tar.extractall(path=FOLDER + "/source")

#                 with tarfile.open(FOLDER + "/target.tar", "r") as tar:
#                     tar.extractall(path=FOLDER + "/target")

#                 # load datasets separately
#                 source_data = datasets.ImageFolder(FOLDER + "/source", transform=transform_steps)
#                 target_data = datasets.ImageFolder(FOLDER + "/target", transform=transform_steps)
#                 if split_data:
#                     return split_train_val_dataset(
#                         target_data, val_data_size=val_data_size, manual_seed=manual_seed
#                     )

#             else:
#                 if not os.path.exists(FOLDER + "/test.tar"):
#                     _ = urllib.request.urlretrieve(
#                         BASE_PATH + "/test.tar", FOLDER + "/test.tar"
#                     )
#         case _:
#             # dataset_name is path
#             raise NotImplementedError(f"Loading of benchmark {benchmark_name} is not implemented yet.")


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

    if square_expected and (sample.shape[1] != sample.shape[2]):
        raise ValueError("Expected square images")

    return int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2])


# !!! If set to None/0 -> no augmentation applied (not doubled dataset)
@overload
def load_augmented_dataset(
    dataset_name: str,
    *,
    split_data: Literal[False],
    train_op: bool = True,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    augment_ops: AugmentOps = AugmentOps(),
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
    augment_ops: AugmentOps = AugmentOps(),
) -> DoubleDataset: ...


def load_augmented_dataset(
    dataset_name: str,
    *,
    split_data: bool = False,
    train_op: bool = True,
    download: bool = True,
    val_data_size: float = 0.4,
    manual_seed: int = 42,
    augment_ops: AugmentOps = AugmentOps(),
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

    original_data = load_dataset(
        dataset_name,
        train_op=train_op,
        split_data=split_data,
        download=download,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
    )

    ref_ds = original_data
    if split_data:
        ref_ds = original_data[0]

    # img height and width are same
    num_channels, img_size, _ = get_dataset_chw(
        ref_ds, square_expected=True  # type: ignore
    )

    transform_steps = create_augmentation_steps(
        img_size, num_channels=num_channels, augment_ops=augment_ops
    )

    augmented_data = load_dataset(
        dataset_name,
        train_op=train_op,
        download=False,
        split_data=split_data,
        transform_steps=transform_steps,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
    )

    if not split_data:
        assert not isinstance(original_data, tuple)
        assert not isinstance(augmented_data, tuple)

        return ConcatDataset([original_data, augmented_data])

    orig_train, orig_val = original_data
    augm_train, _ = augmented_data

    train_data: Dataset[Any] = ConcatDataset([orig_train, augm_train])

    return train_data, orig_val


def get_train_val_loaders(
    first_domain_name: str,
    second_domain_name: str,
    supervised: bool,
    *,
    manual_seed: int = 42,
    val_data_size: float = 0.4,
    augment_ops: AugmentOps = AugmentOps(),
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = False,
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

    first_train, first_val = load_augmented_dataset(
        first_domain_name,
        train_op=True,
        split_data=True,
        augment_ops=augment_ops,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
    )
    second_train, second_val = load_augmented_dataset(
        second_domain_name,
        train_op=True,
        split_data=True,
        augment_ops=augment_ops,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
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
        drop_last=True,  # To ensure consistent batch sizes during training
    )
    val_loader = get_data_loader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        drop_last=False,  # Getting all data in validation
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
    augment_ops: AugmentOps = AugmentOps(),
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
    augment_ops: AugmentOps = AugmentOps(),
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
            pin_memory=pin_memory,
        )

    first_data = load_augmented_dataset(
        first_domain_name,
        train_op=True,
        split_data=False,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        augment_ops=augment_ops,
    )
    second_data = load_augmented_dataset(
        second_domain_name,
        train_op=True,
        split_data=False,
        manual_seed=manual_seed,
        val_data_size=val_data_size,
        augment_ops=augment_ops,
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
        train_op=False,
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
