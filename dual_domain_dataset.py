"""
Module is implementing class to handle two corresponding datasets
 of different sizes. Corresponding datasets work
 with images in supervised and unsupervised manner.
In supervised manner, it ensures that images with the same labels are paired together.
"""

from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset


class DualDomainDataset(Dataset[Tuple[Tensor, int, Tensor, int]]):
    """
    Dataset class to handle two corresponding datasets of different sizes
    in unsupervised manner (no label pairing).
    """

    def __init__(
        self,
        first_data: Dataset[Tuple[Tensor, int]],
        second_data: Dataset[Tuple[Tensor, int]],
    ) -> None:
        """Initialize the dual domain dataset.

        Args:
            first_data (Dataset[Tuple[Tensor, int]]): The first dataset.
            second_data (Dataset[Tuple[Tensor, int]]): The second dataset.
        """
        self.first_data: Dataset[Tuple[Tensor, int]] = first_data
        self.second_data: Dataset[Tuple[Tensor, int]] = second_data

        # Pylance gives an error here, but len() works for Dataset
        self.first_size: int = len(self.first_data)  # type: ignore
        self.second_size: int = len(self.second_data)  # type:ignore

        if self.first_size == 0 or self.second_size == 0:
            raise ValueError("Datasets must not be empty.")

        self.max_size: int = max(self.first_size, self.second_size)

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

        first_img: Tensor
        first_label: int
        first_img, first_label = self.first_data[first_idx]

        second_img: Tensor
        second_label: int
        second_img, second_label = self.second_data[second_idx]

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
    ) -> None:
        """Initialize the dual domain supervised dataset.

        Args:
            first_data (Dataset[Tuple[Tensor, int]]): The first dataset.
            second_data (Dataset[Tuple[Tensor, int]]): The second dataset.

        Raises:
            ValueError: If the first dataset is smaller than the second one.
            KeyError: If a label in the second dataset is not present in the first dataset.
            KeyError: If a label in the second dataset is missing.
        """
        super().__init__(first_data, second_data)

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
) -> DualDomainDataset:
    """
    Combine two datasets into a single dual domain dataset.

    Args:
        first_data (Dataset[Tuple[Tensor, int]]): The first domain dataset.
        second_data (Dataset[Tuple[Tensor, int]]): The second domain dataset.
        supervised (bool, optional): Whether to use supervised learning. Defaults to False.

    Returns:
        DualDomainDataset: The dual domain dataset.
    """
    if supervised:
        dual_data = DualDomainSupervisedDataset(first_data, second_data)

    else:
        dual_data = DualDomainDataset(first_data, second_data)

    return dual_data
