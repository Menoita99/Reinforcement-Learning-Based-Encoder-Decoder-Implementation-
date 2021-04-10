import torch
from torch import nn


class PolicyNet(nn.Module):

    def __init__(self, input_dim,hidden_dim, output_dim):

        self.net = nn.Sequential(
            nn.linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.linear(hidden_dim,output_dim),
        )

        def forward(self, input):
            return self.net(input)