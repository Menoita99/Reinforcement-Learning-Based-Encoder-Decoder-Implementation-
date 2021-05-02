from torch import nn


class Mlp(nn.Module):

    def __init__(self, input_dim, features_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(num_features=128),
            nn.Linear(128, features_dim),
        )

    def forward(self, input):
        self.train()
        if len(input) == 4:
            input = input.unsqueeze(0)
            self.eval()
        return self.net(input)


class Cnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(  # dim 4 x 4
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding=1),  # output: 8 x 4 x 4
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1),  # output: 16 x 4 x 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 16 x 2 x 2

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1),  # output: 32 x 4 x 4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 32 x 1 x 1

            nn.Flatten(),  # output batch_size x 128 , 128 was obtain experimentally
        )

    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        elif len(input.size()) == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        return self.net(input)


class Gru(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to('cuda')
        return hidden


class CnnGru(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(self).__init__()
        #TODO