import copy
from torch import nn


class PolicyNet(nn.Module):

    def __init__(self, encoder, input_dim, hidden_dim, output_dim, device):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(device=device)

        if encoder is not None:
            self.policy = nn.Sequential(
                encoder,
                self.policy)

        self.target = copy.deepcopy(self.policy).to(device=device)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "policy":
            return self.policy(input)
        elif model == "target":
            return self.target(input)
