from torch.utils.data import Dataset
import torch


class DualDomainDataset(Dataset):
    def __init__(self, first_data, second_data):
        self.first_data = first_data
        self.second_data = second_data

        self.first_size = len(self.first_data)
        self.second_size = len(self.second_data)

        self.first_labels = self._extract_labels(self.first_data)
        self.second_labels = self._extract_labels(self.second_data)

        self.max_size = max(self.first_size, self.second_size)

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        first_idx = idx % self.first_size
        second_idx = idx % self.second_size

        first_img, first_label = self.first_data[first_idx]
        second_img, second_label = self.second_data[second_idx]

        if not isinstance(first_label, torch.Tensor):
            first_label = torch.tensor(first_label)

        if not isinstance(second_label, torch.Tensor):
            second_label = torch.tensor(second_label)

        return first_img, first_label, second_img, second_label

    @staticmethod
    def _extract_labels(data):
        labels = []

        for _, label in data:
            labels.append(label)

        labels = torch.tensor(labels)

        return labels

