from torch.utils.data import Dataset
import torch
import random


class DualDomainDataset(Dataset):
    def __init__(self, first_data, second_data):
        self.first_data = first_data
        self.second_data = second_data

        self.first_size = len(self.first_data)
        self.second_size = len(self.second_data)

        self.max_size = max(self.first_size, self.second_size)

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        first_idx = idx % self.first_size
        second_idx = idx % self.second_size

        first_img, first_label = self.first_data[first_idx]
        second_img, second_label = self.second_data[second_idx]

        return first_img, first_label, second_img, second_label

    @staticmethod
    def _extract_labels(data):
        labels = []

        for _, label in data:
            labels.append(label)

        labels = torch.tensor(labels)

        return labels


class DualDomainSupervisedDataset(DualDomainDataset):  # first dataset should be always the first one -> raise error
    def __init__(self, first_data, second_data):
        super().__init__(first_data, second_data)

        if self.first_size < self.second_size:
            raise ValueError("First dataset cannot be smaller than the second one.")

        self.first_labels = self._extract_labels(self.first_data)
        self.second_labels = self._extract_labels(self.second_data)

        self.unique_labels = torch.unique(self.first_labels).tolist()

        self.label_indices = {label: [] for label in self.unique_labels}  # for every target get list of indices

        for idx, label in enumerate(self.second_labels):
            self.label_indices[label.item()].append(idx)

        self.label_usage_counts = dict()
        self.weights = dict()
        for label, indices in self.label_indices.items():
            self.label_usage_counts[label] = {idx: 0 for idx in indices}
            self.weights[label] = {idx: 1 for idx in indices}

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        first_idx = idx % self.first_size
        first_img, first_label = self.first_data[first_idx]

        indices = self.label_indices[first_label]
        usage_counts = self.label_usage_counts[first_label]

        second_idx = random.choices(indices, weights=self.weights[first_label], k=1)[0]

        second_img, second_label = self.second_data[second_idx]

        # update current position for the label for second data
        usage_counts[second_idx] += 1
        self.weights[first_label][second_idx] = 1.0 / (1 + usage_counts[second_idx])
        if first_label != second_label:
            raise AssertionError(f'Labels should be the same. First label: {first_label}. Second label: {second_label}')

        return first_img, first_label, second_img, second_label


def custom_collate_fn(batch):
    # Unpack the batch into separate components
    first_imgs, first_labels, second_imgs, second_labels = zip(*batch)

    # Stack the images and labels
    first_imgs = torch.stack(first_imgs)
    first_labels = torch.tensor(first_labels)

    second_imgs = torch.stack(second_imgs)
    second_labels = torch.tensor(second_labels)

    return first_imgs, first_labels, second_imgs, second_labels


def get_dual_domain_dataset(first_data, second_data, supervised):
    if supervised:
        dual_data = DualDomainSupervisedDataset(first_data, second_data)

    else:
        dual_data = DualDomainDataset(first_data, second_data)

    return dual_data
