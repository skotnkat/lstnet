from torchvision import datasets
from torchvision.transforms.v2 import Compose, RandomRotation, RandomAffine, RandomResizedCrop, ToImage, ToDtype
from torch.utils.data import ConcatDataset, DataLoader
import torch
from dual_domain_dataset import DualDomainDataset, DualDomainSupervisedDataset

from models import lstnet
import utils

# toTensor, will be deprecated from future version
BASIC_TRANSFORMATION = Compose([
    ToImage(),
    ToDtype(torch.float32, scale=True)
])


def create_augmentation_steps(img_size):
    dy_translation = dx_translation = 2 / img_size
    return Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        RandomAffine(
            degrees=(-10, 10),
            translate=(dx_translation, dy_translation),
            scale=(0.9, 1.1)
        )
    ])


def load_dataset(dataset_name, train_op=True, transform_steps=BASIC_TRANSFORMATION, **kwargs):
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

    data_loader = DataLoader(dual_data, batch_size=utils.BATCH_SIZE, shuffle=True)

    print(f'Obtained Data Loader')
    return data_loader
