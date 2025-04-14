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


class DualDomainSupervisedDataset(DualDomainDataset): # first dataset should be always the first one -> raise error
    def __init__(self, first_data, second_data):
        super().__init__(first_data, second_data)

        if self.first_size < self.second_size:
            raise ValueError("First dataset cannot be smaller than the second one.")

        self.unique_labels = torch.unique(self.first_labels).tolist()

        self.label_indices = {label: [] for label in self.unique_labels}  # for every target get list of indices

        for idx, label in enumerate(self.second_labels):
            self.label_indices[label.item()].append(idx)

        self.label_cur_pos = {label: 0 for label in self.unique_labels}
        self.label_max_size = {label: len(self.label_indices[label])-1 for label in self.unique_labels}

    def __len__(self):
        return self.max_size

    def __getitem__(self, idx):
        first_idx = idx % self.first_size
        first_img, first_label = self.first_data[first_idx]

        cur_label_pos = self.label_cur_pos[first_label]
        second_idx = self.label_indices[first_label][cur_label_pos]

        second_img, second_label = self.second_data[second_idx]

        # update current position for the label for second data
        self.label_cur_pos[first_label] = (cur_label_pos + 1) % self.label_max_size[first_label]

        if first_label != second_label:
            raise AssertionError(f'Labels should be the same. First label: {first_label}. Second label: {second_label}')

        if not isinstance(first_label, torch.Tensor):
            first_label = torch.tensor(first_label)

        if not isinstance(second_label, torch.Tensor):
            second_label = torch.tensor(second_label)

        return first_img, first_label, second_img, second_label


def custom_collate_fn(batch):
    # Unpack the batch into separate components
    first_imgs, first_labels, second_imgs, second_labels = zip(*batch)

    for idx in range(len(second_imgs)):
        img = second_imgs[idx]

    # Stack the images and labels
    first_imgs = torch.stack(first_imgs)
    first_labels = torch.stack(first_labels)
    second_imgs = torch.stack(second_imgs)
    second_labels = torch.stack(second_labels)

    return first_imgs, first_labels, second_imgs, second_labels
