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
    return Compose([
        RandomRotation(10),
        RandomAffine(0, translate=(0.02, 0.02)),  # 2% of 28x28 ≈ 0.56 pixels
        RandomResizedCrop(img_size, scale=(0.9, 1.1)),
        ToImage(),
        ToDtype(torch.float32, scale=True)
    ])


def load_dataset(dataset_name, train_op=True, transform_steps=BASIC_TRANSFORMATION, **kwargs):
    if dataset_name == 'MNIST':
        data = datasets.MNIST(root="./data", train=train_op, transform=transform_steps, **kwargs)
        print('Loaded MNIST')

    elif dataset_name == 'USPS':
        data = datasets.USPS(root="./data", train=train_op, transform=transform_steps, **kwargs)
        print('Loaded USPS')

    elif dataset_name == 'SVHN':
        # targets are called labels and are array
        split = 'test'
        if train_op:
            split = 'train'
        data = datasets.SVHN(root="./data", split=split, transform=transform_steps, **kwargs)
        
    else:
        raise ValueError(f"Not able to load dataset {dataset_name}")

    return data

