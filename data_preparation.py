from torchvision import datasets
from torchvision.transforms.v2 import Compose, RandomAffine, ToImage, ToDtype, Normalize
from torch.utils.data import ConcatDataset, DataLoader
import torch
from dual_domain_dataset import DualDomainDataset, DualDomainSupervisedDataset, custom_collate_fn

from models import lstnet
import utils

BASIC_TRANSFORMATION = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True),  # scale from [0, 250] to [0, 1]
    Normalize(mean=[0.5], std=[0.5]),  # scales from [0, 1] to [-1, 1]
])


def create_augmentation_steps(img_size):
    dy_translation = dx_translation = 2 / img_size
    return Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5], std=[0.5]),
        RandomAffine(
            degrees=(-10, 10),
            translate=(dx_translation, dy_translation),
            scale=(0.9, 1.1)
        ),
    ])

# option to pass dataset_path instead of name and flag to load from file

def load_dataset(dataset_name, train_op=True, transform_steps=BASIC_TRANSFORMATION, **kwargs):
    # load data from torchvision datasets
    if dataset_name == 'MNIST':
        data = datasets.MNIST(root="./data", train=train_op, transform=transform_steps, **kwargs)

    elif dataset_name == 'USPS':
        data = datasets.USPS(root="./data", train=train_op, transform=transform_steps, **kwargs)

    elif dataset_name == 'SVHN':
        # targets are called labels and are array
        split = 'test'
        if train_op:
            split = 'train'
        data = datasets.SVHN(root="./data", split=split, transform=transform_steps, **kwargs)
        
    else:
        raise ValueError(f"Not able to load dataset {dataset_name}")

    return data


def load_augmented_dataset(dataset_name, train_op=True, download=True):
    print(f'Loading dataset: {dataset_name}')
    original_data = load_dataset(dataset_name, train_op=train_op, download=download)

    img_size = original_data[0][0].shape[1]  # size to use for resize
    transform_steps = create_augmentation_steps(img_size)

    augmented_data = load_dataset(dataset_name, train_op=train_op, download=False, transform_steps=transform_steps)

    return ConcatDataset([original_data, augmented_data])


def get_training_loader(first_domain_name, second_domain_name, supervised=True):
    first_data = load_augmented_dataset(first_domain_name, train_op=True)
    second_data = load_augmented_dataset(second_domain_name, train_op=True)

    if len(first_data) < len(second_data):
        raise ValueError("First dataset should be larger.")

    if supervised:
        dual_data = DualDomainSupervisedDataset(first_data, second_data)

    else:
        dual_data = DualDomainDataset(first_data, second_data)

    first_img, _, second_img, _ = dual_data.__getitem__(0)

    lstnet.FIRST_INPUT_SHAPE = first_img.shape[1:]
    lstnet.FIRST_IN_CHANNELS_NUM = first_img.shape[0]

    lstnet.SECOND_INPUT_SHAPE = second_img.shape[1:]
    lstnet.SECOND_IN_CHANNELS_NUM = second_img.shape[0]

    pin_memory = utils.DEVICE != "cpu"  # locking in physical RAM, higher data transfer with gpu
    data_loader = DataLoader(dual_data, batch_size=utils.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn,
                             pin_memory=pin_memory, num_workers=utils.NUM_WORKERS, persistent_workers=True)

    print(f'Obtained Data Loader')
    return data_loader


def get_testing_loader(domain_name):
    data = load_dataset(domain_name, train_op=False, shuffle=False)
    data_loader = DataLoader(data, batch_size=utils.BATCH_SIZE, num_workers=utils.NUM_WORKERS)

    return data_loader
