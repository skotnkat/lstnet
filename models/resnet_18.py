import torch.nn as nn
from extended_layers import Conv2dExtended, MaxPool2dExtended


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2dExtended(
            in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding="same",
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = Conv2dExtended(
            out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # projection to map it into correct shape
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, in_channels_num=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels_num
        self.out_channels = 64
        self.num_classes = num_classes

        first_layer = self._get_first_layer()

        block1 = self._get_stand_layer(64, 64)
        block2 = self._get_stand_layer(64, 128)
        block3 = self._get_stand_layer(128, 256)
        block4 = self._get_stand_layer(256, 512)

        last_layer = self._get_last_layer()

        self.layers = nn.Sequential(
            first_layer, block1, block2, block3, block4, last_layer
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def _get_first_layer(self):
        conv1 = Conv2dExtended(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=7,
            stride=2,
            padding="same",
        )
        bn = nn.BatchNorm2d(self.out_channels)
        relu = nn.ReLU()
        max_pool = MaxPool2dExtended(kernel_size=3, stride=2, padding="same")

        return nn.Sequential(conv1, bn, relu, max_pool)

    def _get_last_layer(self):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, self.num_classes)
        )

    def _get_stand_layer(self, in_channels, out_channels):
        stride = 2
        if in_channels == out_channels:
            stride = 1

        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )
