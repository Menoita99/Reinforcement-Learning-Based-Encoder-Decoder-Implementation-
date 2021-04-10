import copy
from torch import nn


class PolicyNet(nn.Module):

    def __init__(self, input_dim,hidden_dim, output_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim),
        )

        self.target = copy.deepcopy(self.policy)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "policy":
            return self.policy(input)
        elif model == "target":
            return self.target(input)

