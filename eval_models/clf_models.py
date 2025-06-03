import torch.nn as nn
from models.discriminator import Discriminator
from models.extended_layers import MaxPool2dExtended, Conv2dExtended
import torch.optim as optim


class BaseClf(Discriminator):
    def __init__(self, input_size, in_channels, params, optimizer=None, epochs=50, patience=5, negative_slope=0.01):
        self.input_size = tuple(input_size)
        self.in_channels_num = in_channels

        super().__init__(self.input_size, self.in_channels_num, params, negative_slope=negative_slope)

        self.criterion = nn.CrossEntropyLoss()

        # default parameters can be overwritten
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience


class MnistClf(Discriminator):
    def __init__(self, params):
        self.input_size = (28, 28)
        self.in_channels_num = 1
        super().__init__(self.input_size, self.in_channels_num, params)

        self.criterion = nn.CrossEntropyLoss()

        # default parameters can be overwritten
        self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
        self.epochs = 40
        self.patience = 5

    def _create_last_layer(self):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dense_layer_params["dropout_p"]),
            nn.Linear(in_features=self.dense_layer_params["in_features"],
                      out_features=self.dense_layer_params["out_features"])
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

    def _create_last_layer(self):
        last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dense_layer_params["dropout_p"]),
            nn.Linear(in_features=self.dense_layer_params["in_features"],
                      out_features=self.dense_layer_params["out_features"])
        )

        return last_layer


# class SvhnClf(Discriminator):
#     def __init__(self, params):
#         self.input_size = (32, 32)
#         self.in_channels_num = 3
#         self.global_params = params[-1]
#         super().__init__(self.input_size, self.in_channels_num, params[:-1])
#
#         self.optimizer = optim.Adam(self.layers.parameters(), lr=0.001)
#         self.criterion = nn.CrossEntropyLoss()
#         self.epochs = 50
#         self.patience = 10
#
#
#     def _create_stand_layer(self, params, in_channels, input_size):
#         conv_params, pool_params = params
#         conv = Conv2dExtended(in_channels, input_size=input_size, **conv_params)
#         batch_norm = nn.BatchNorm2d(conv_params["out_channels"], momentum=self.global_params["batch_norm_momentum"])
#         relu = nn.ReLU()
#
#         layers = [conv, batch_norm, relu]
#
#         if len(pool_params):
#             output_size = conv.compute_output_size(input_size)
#             layers.append(MaxPool2dExtended(input_size=output_size, **pool_params))
#             layers.append(negative_slope=self.global_params["leaky_relu_neg_slope"])
#
#         return nn.Sequential(*layers)
#
#     def _create_last_layer(self):
#         last_layer = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(self.dense_layer_params["dropout_p"]),
#             nn.Linear(in_features=self.dense_layer_params["in_features"],
#                       out_features=self.dense_layer_params["out_features"])
#         )
#
#         return last_layer
