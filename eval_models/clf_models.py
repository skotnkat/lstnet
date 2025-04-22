import torch.nn as nn
from models.discriminator import Discriminator
from models.extended_layers import MaxPool2dExtended, Conv2dExtended
import torch.optim as optim


class MnistClf(Discriminator):
    def __init__(self, params):
        self.input_size = (28, 28)
        self.in_channels_num = 1
        super().__init__(self.input_size, self.in_channels_num, params)

        self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 40
        self.patience = 5

    @staticmethod
    def _crete_last_layer(params):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(**params)
        )

        return last_layer

