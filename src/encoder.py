import torch
from torch import nn


class Mlp(nn.Module):

    def __init__(self, input_dim, features_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.BatchNorm1d(num_features=20),
            nn.Linear(128, features_dim),
        )

    def forward(self, input):
        return self.net(input)


class Cnn(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.net = nn.Sequential(
            # TODO
        )

        def forward(self, input):
            return self.net(input)


class Gru(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.net = nn.Sequential(
            # TODO
        )

        def forward(self, input):
            return self.net(input)


class CnnGru(nn.Module):

    def __init__(self, input_dim, output_dim):
        self.net = nn.Sequential(
            # TODO
        )

        def forward(self, input):
            return self.net(input)
