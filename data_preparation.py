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

