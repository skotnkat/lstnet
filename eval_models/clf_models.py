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


class UspsClf(Discriminator):
    def __init__(self, params):
        self.input_size = (16, 16)
        self.in_channels_num = 1
        super().__init__(self.input_size, self.in_channels_num, params)

        self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 40
        self.patience = 10

    @staticmethod
    def _crete_last_layer(params):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(**params)
        )

        return last_layer


class SvhnClf(Discriminator):
    def __init__(self, params):
        self.input_size = (16, 16)
        self.in_channels_num = 1
        super().__init__(self.input_size, self.in_channels_num, params)

        self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 50
        self.patience = 10

    @staticmethod
    def _create_stand_layer(params, in_channels, input_size):
        conv_params, pool_params = params
        conv = Conv2dExtended(in_channels, input_size=input_size, **conv_params)
        batch_norm = nn.BatchNorm2d(params["out_channels"], momentum=0.01, eps=0.001)
        relu = nn.ReLU()

        layers = [conv, batch_norm, relu]

        if len(pool_params):
            output_size = conv.compute_output_size(input_size)
            layers.append(MaxPool2dExtended(input_size=output_size, **pool_params))
            layers.append(nn.Dropout(0.3))

        return nn.Sequential(*layers)

    @staticmethod
    def _crete_last_layer(params):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(**params)
        )

        return last_layer
