"""
Module is implementing class to handle two corresponding datasets
 of different sizes. Corresponding datasets work
 with images in supervised and unsupervised manner.
In supervised manner, it ensures that images with the same labels are paired together.
"""

from typing import Tuple, List, Dict, Optional, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose


class DualDomainDataset(Dataset[Tuple[Tensor, int, Tensor, int]]):
    """
    Dataset class to handle two corresponding datasets of different sizes
    in unsupervised manner (no label pairing).
    """

    def __init__(
        self,
        first_data: Dataset[Tuple[Tensor, int]],
        second_data: Dataset[Tuple[Tensor, int]],
        first_transform: Optional[Compose] = None,
        second_transform: Optional[
            Compose
        ] = None,  # Augmentation is skipped when None is apssed
    ) -> None:
        """Initialize the dual domain dataset.

        Args:
            first_data (Dataset[Tuple[Tensor, int]]): The first dataset.
            second_data (Dataset[Tuple[Tensor, int]]): The second dataset.
            repeat_transform (Optional[Compose], optional): Transform to apply to augmented
                version of the bigger dataset. If provided, the bigger dataset will be
                doubled (original + augmented). Defaults to None.
        """
        # Get original sizes
        first_size_original: int = len(first_data)  # type: ignore
        second_size_original: int = len(second_data)  # type: ignore

        if first_size_original == 0 or second_size_original == 0:
            raise ValueError("Datasets must not be empty.")

        # Determine which dataset is smaller
        first_is_smaller = first_size_original < second_size_original

        # If repeat_transform is provided, augment the bigger dataset (double its size)
        if first_is_smaller:
            self.first_data = first_data
            self.repeat_transform = first_transform
            if second_transform is not None:
                augmented_second = AugmentedDataset(second_data, second_transform)
                self.second_data = ConcatDataset([second_data, augmented_second])
            else:
                self.second_data = second_data

        else:
            self.second_data = second_data
            self.repeat_transform = second_transform

            if first_transform is not None:
                augmented_first = AugmentedDataset(first_data, first_transform)
                self.first_data = ConcatDataset([first_data, augmented_first])
            else:
                self.first_data = first_data

        # Update sizes after potential augmentation
        self.first_size: int = len(self.first_data)  # type: ignore
        self.second_size: int = len(self.second_data)  # type: ignore
        self.max_size: int = max(self.first_size, self.second_size)

        # Track which is smaller and store transform for repeat cycles
        self.first_is_smaller = self.first_size < self.second_size
        self.smaller_size = min(self.first_size, self.second_size)

    def __len__(self) -> int:
        """Return the maximum size of the two datasets."""
        return self.max_size

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor, int]:
        """Get a pair of images and their labels from the two datasets.

        Args:
            idx (int): The index of the data point.

        Returns:
            Tuple[Tensor, int, Tensor, int]:
                A tuple containing the images and their labels from both datasets.
        """
        first_idx: int = idx % self.first_size
        second_idx: int = idx % self.second_size

        # Check if we're in a repeat cycle (beyond the smaller dataset's size)
        repeat_flag: bool = (idx >= self.smaller_size) and (
            self.repeat_transform is not None
        )

        first_img: Tensor
        first_label: int
        first_img, first_label = self.first_data[first_idx]

        second_img: Tensor
        second_label: int
        second_img, second_label = self.second_data[second_idx]

        # Apply repeat_transform to the smaller dataset when it cycles
        if repeat_flag:
            if self.first_is_smaller:
                first_img = self.repeat_transform(first_img)
            else:
                second_img = self.repeat_transform(second_img)

        # Temporary type check for different dataset implementations
        if not isinstance(first_label, int) or not isinstance(second_label, int):
            raise TypeError("Labels must be of type int.")

        return first_img, first_label, second_img, second_label

    @staticmethod
    def _extract_labels(data: Dataset[Tuple[Tensor, int]]) -> Tensor:
        """Extract labels from the dataset, if they are already not in tensor form as an attribute.

        Args:
            data (Dataset[Tuple[Tensor, int]]): The dataset to extract labels from.

        Returns:
            Tensor: A tensor containing the labels.
        """
        for attr in ["targets", "labels"]:
            if hasattr(data, attr):
                return torch.as_tensor(getattr(data, attr), dtype=torch.long)

        labels: List[int] = []
        for _, label in data:
            labels.append(label)

        labels_tensor: Tensor = torch.tensor(labels, dtype=torch.long)
        return labels_tensor

    def get_input_dims(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return raw tensor shapes as (C, H, W) for both datasets."""
        first_shape = tuple(self.first_data[0][0].shape)
        second_shape = tuple(self.second_data[0][0].shape)

        # Cast to plain ints to satisfy type hints (Tensor dims can be torch.SymInt)
        c1, h1, w1 = first_shape[-3:]
        c2, h2, w2 = second_shape[-3:]
        first_dims: Tuple[int, int, int] = (int(c1), int(h1), int(w1))
        second_dims: Tuple[int, int, int] = (int(c2), int(h2), int(w2))

        return first_dims, second_dims


class DualDomainSupervisedDataset(DualDomainDataset):
    """
    Dataset class to handle two datasets of different sizes that have corresponding labels.
    Merges them in a supervised manner (with label pairing).
    """

    # first dataset should be always the first one -> raise error
    def __init__(
        self,
        first_data: Dataset[Tuple[Tensor, int]],
        second_data: Dataset[Tuple[Tensor, int]],
        first_transform: Optional[Compose] = None,
        second_transform: Optional[Compose] = None,
    ) -> None:
        """Initialize the dual domain supervised dataset.

        Args:
            first_data (Dataset[Tuple[Tensor, int]]): The first dataset.
            second_data (Dataset[Tuple[Tensor, int]]): The second dataset.
            first_transform (Optional[Compose], optional): Transform for first dataset. Defaults to None.
            second_transform (Optional[Compose], optional): Transform for second dataset. Defaults to None.

        Raises:
            ValueError: If the first dataset is smaller than the second one.
            KeyError: If a label in the second dataset is not present in the first dataset.
            KeyError: If a label in the second dataset is missing.
        """
        super().__init__(first_data, second_data, first_transform, second_transform)

        if self.first_size < self.second_size:
            raise ValueError("First dataset cannot be smaller than the second one.")

        first_labels: Tensor = self._extract_labels(self.first_data)
        second_labels: Tensor = self._extract_labels(self.second_data)
        second_indices_shuffled: List[int] = torch.randperm(len(self.second_data)).tolist()  # type: ignore

        unique_labels: List[int] = torch.unique(first_labels).tolist()  # type: ignore

        label_indices: Dict[int, List[int]] = {
            label: [] for label in unique_labels
        }  # for every target get list of indices

        for idx in second_indices_shuffled:
            label = second_labels[idx]

            try:
                label_indices[int(label.item())].append(idx)
            except KeyError as e:
                raise KeyError(
                    f"Label {label} in the second dataset is not present in the first dataset. \
                        Caught Exception: {e}"
                ) from e

        # Check if any label is missing
        for label, indices in label_indices.items():
            if len(indices) == 0:
                raise KeyError(f"Label {label} is missing in the second dataset.")

        label_pointers: Dict[int, int] = {label: 0 for label in unique_labels}
        rank: List[int] = []

        for label in first_labels:
            label = int(label.item())
            pos = label_pointers[label]
            second_idx_for_label: int = label_indices[label][pos]

            rank.append(second_idx_for_label)

            label_pointers[label] = (pos + 1) % len(label_indices[label])

        self.second_rank = torch.tensor(rank, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor, int]:
        """
        Get a pair of images and their labels from both datasets.
        Both images will have the same label.

        Args:
            idx (int): The index of the sample to retrieve.

        Raises:
            AssertionError: If the labels of the two images do not match (should never happen).

        Returns:
            Tuple[Tensor, int, Tensor, int]: A tuple containing the images and their labels.
        """
        first_idx: int = idx % self.first_size

        first_img: Tensor
        first_label: int

        first_img, first_label = self.first_data[first_idx]

        second_idx = self.second_rank[first_idx].item()
        second_img: Tensor
        second_label: int
        second_img, second_label = self.second_data[second_idx]

        repeat_flag: bool = (idx >= self.smaller_size) and (self.repeat_transform is not None)
        # Temporary type check for different dataset implementations
        if not isinstance(first_label, int) or not isinstance(second_label, int):
            raise TypeError("Labels must be of type int.")

        if first_label != second_label:
            raise AssertionError(
                f"Labels should be the same. First label: {first_label}. \
                    Second label: {second_label}"
            )

        return first_img, first_label, second_img, second_label


def custom_collate_fn(
    batch: List[Tuple[Tensor, int, Tensor, int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Custom collate function for DataLoader.

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        tuple: A tuple containing stacked images and labels for both domains.
    """
    # Unpack the batch into separate components
    first_imgs, first_labels, second_imgs, second_labels = zip(*batch)

    # Stack the images and labels
    first_imgs = torch.stack(first_imgs)
    first_labels = torch.as_tensor(first_labels, dtype=torch.long)

    second_imgs = torch.stack(second_imgs)
    second_labels = torch.as_tensor(second_labels, dtype=torch.long)

    return first_imgs, first_labels, second_imgs, second_labels


def get_dual_domain_dataset(
    first_data: Dataset[Tuple[Tensor, int]],
    second_data: Dataset[Tuple[Tensor, int]],
    supervised: bool = False,
    first_transform: Optional[Compose] = None,
    second_transform: Optional[Compose] = None,
) -> DualDomainDataset:
    """
    Combine two datasets into a single dual domain dataset.

    Args:
        first_data (Dataset[Tuple[Tensor, int]]): The first domain dataset.
        second_data (Dataset[Tuple[Tensor, int]]): The second domain dataset.
        supervised (bool, optional): Whether to use supervised learning. Defaults to False.
        first_transform (Optional[Compose], optional): Transform to apply to first dataset augmentation.
            Defaults to None.
        second_transform (Optional[Compose], optional): Transform to apply to second dataset augmentation.
            Defaults to None.

    Returns:
        DualDomainDataset: The dual domain dataset.
    """
    if supervised:
        dual_data = DualDomainSupervisedDataset(
            first_data, second_data, first_transform, second_transform
        )

    else:
        dual_data = DualDomainDataset(
            first_data, second_data, first_transform, second_transform
        )

    return dual_data


class AugmentedDataset(Dataset[Tuple[Tensor, int]]):
    """Wrapper to apply transform to an existing dataset."""

    def __init__(
        self, base_dataset: Dataset[Tuple[Tensor, int]], transform: Optional[Compose]
    ):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def augment_dataset(dataset, transform) -> Dataset[Any]:
    pass
