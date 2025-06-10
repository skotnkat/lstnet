from torchvision import datasets
from torchvision.transforms.v2 import Compose, RandomAffine, ToImage, ToDtype, Normalize
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torch
from dual_domain_dataset import get_dual_domain_dataset, custom_collate_fn

import utils

BASIC_TRANSFORMATION = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),  # scale from [0, 250] to [0, 1]
    Normalize(mean=[0.5], std=[0.5]),  # scales from [0, 1] to [-1, 1]
])


def create_augmentation_steps(img_size, rotation=10, zoom=0.1, shift=2):
    dy_translation = dx_translation = shift / img_size
    return Compose([
        ToImage(),
        RandomAffine(
            degrees=(-rotation, rotation),
            translate=(dx_translation, dy_translation),
            scale=(1-zoom, 1+zoom)
        ),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5], std=[0.5]),
    ])


def load_dataset(dataset_name, train_op=True, transform_steps=BASIC_TRANSFORMATION, download=True, split_data=False, **kwargs):
    # load data from torchvision datasets
    if dataset_name == 'MNIST':
        data = datasets.MNIST(root="./data", train=train_op, transform=transform_steps, download=download, **kwargs)

    elif dataset_name == 'USPS':
        data = datasets.USPS(root="./data", train=train_op, transform=transform_steps, download=download, **kwargs)

    elif dataset_name == 'SVHN':
        # targets are called labels and are array
        split = 'test'
        if train_op:
            split = 'train'
        data = datasets.SVHN(root="./data", split=split, transform=transform_steps, download=download, **kwargs)

    else:
        print(f'Trying to load dataset {dataset_name} locally')
        data = torch.load(dataset_name, weights_only=False)  # dataset_name is path
        print(f'Dataset loaded, number of records: {len(data)}, shape: {data[0][0].shape}')

    if not split_data:
        return data

    g = torch.Generator()
    g.manual_seed(utils.MANUAL_SEED)

    val_size = int(len(data)*utils.VAL_SIZE)
    train_size = len(data) - val_size

    train_data, val_data = random_split(data, [train_size, val_size], generator=g)

    return train_data, val_data


def get_dataset_img_size(dataset):
    single_img = dataset[0][0]
    if isinstance(dataset, tuple):
        single_img = single_img[0]

    return single_img.shape[1]  # size to use for resize


def load_augmented_dataset(dataset_name, train_op=True, download=True, split_data=False, rotation=10, zoom=0.1, shift=2):
    print(f'Loading dataset: {dataset_name}')
    original_data = load_dataset(dataset_name, train_op=train_op, download=download, split_data=split_data)

    img_size = get_dataset_img_size(original_data)
    transform_steps = create_augmentation_steps(img_size, rotation, zoom, shift)

    # use transformations also on original data -> improve robustness
    # original_data = load_dataset(dataset_name, train_op=train_op, download=False, transform_steps=transform_steps,
    #                               split_data=split_data)
    augmented_data = load_dataset(dataset_name, train_op=train_op, download=False, transform_steps=transform_steps, split_data=split_data)

    if not split_data:
        return ConcatDataset([original_data, augmented_data])

    orig_train, orig_val = original_data
    augm_train, _ = augmented_data  # forgetting augmented validation, do not want that

    train_data = ConcatDataset([orig_train, augm_train])

    return train_data, orig_val



def get_train_val_loaders(first_domain_name, second_domain_name, supervised, rotation=10, zoom=0.1, shift=2):
    first_train, first_val = load_augmented_dataset(first_domain_name, train_op=True, split_data=True, rotation=rotation, zoom=zoom, shift=shift)
    second_train, second_val = load_augmented_dataset(second_domain_name, train_op=True, split_data=True, rotation=rotation, zoom=zoom, shift=shift)

    val_data = get_dual_domain_dataset(first_val, second_val, supervised=False)
    train_data = get_dual_domain_dataset(first_train, second_train, supervised)

    utils.set_input_dimensions(train_data)

    pin_memory = utils.DEVICE != "cpu"  # locking in physical RAM, higher data transfer with gpu
    train_loader = DataLoader(train_data, batch_size=utils.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                             pin_memory=pin_memory, num_workers=utils.NUM_WORKERS, persistent_workers=False)
    val_loader = DataLoader(val_data, batch_size=utils.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                             pin_memory=pin_memory, num_workers=utils.NUM_WORKERS, persistent_workers=False)
    print(f'Obtained Data Loader for both training and validation')

    return train_loader, val_loader

def get_training_loader(first_domain_name, second_domain_name, supervised=True, split_data=False, rotation=10, zoom=0.1, shift=2):
    if split_data:
        return get_train_val_loaders(first_domain_name, second_domain_name, supervised)

    first_data = load_augmented_dataset(first_domain_name, train_op=True, rotation=rotation, zoom=zoom, shift=shift)
    second_data = load_augmented_dataset(second_domain_name, train_op=True, rotation=rotation, zoom=zoom, shift=shift)
    dual_data = get_dual_domain_dataset(first_data, second_data, supervised)

    utils.set_input_dimensions(dual_data)

    pin_memory = utils.DEVICE != "cpu"  # locking in physical RAM, higher data transfer with gpu
    data_loader = DataLoader(dual_data, batch_size=utils.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                             pin_memory=False, num_workers=utils.NUM_WORKERS, persistent_workers=False)
    print(f'Obtained Data Loader for training')

    return data_loader


def get_testing_loader(domain_name):
    data = load_dataset(domain_name, train_op=False)
    data_loader = DataLoader(data, batch_size=utils.BATCH_SIZE, shuffle=False, num_workers=utils.NUM_WORKERS)

    return data_loader
