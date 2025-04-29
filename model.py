# model.py
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes, conv_filters, filter_size, pool_size, dense_units):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0]

        # Define 3 convolutional layers with BatchNorm, ReLU, MaxPooling
        for filters in conv_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size=filter_size, padding=1),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_size)
                )
            )
            in_channels = filters

        # Calculate flattened size
        self._initialize_flatten_size(input_shape)

        # Dense layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, num_classes)
        )

    def _initialize_flatten_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
            self.flatten_size = x.view(-1).size(0)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.fc_layers(x)
        return x


def build_cnn_model(input_shape, num_classes, conv_filters, filter_size, pool_size, dense_units):
    model = CNNModel(input_shape, num_classes, conv_filters, filter_size, pool_size, dense_units)
    return model