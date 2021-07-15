import torch
from torch import nn


class Mlp(nn.Module):

    def __init__(self, input_dim, features_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, features_dim),
        )

    def forward(self, input):
        self.train()
        if len(input) == 5:
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
                           # without MaxPool = 1568
        )

    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        elif len(input.size()) == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        return self.net(input)


class Gru(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,dropout):
        super(Gru, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_hidden_dim = 10

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,dropout=dropout).cuda()
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        out, _ = self.gru(x, (h0.detach())) #out, (hn, cn)
        out = self.fc(out[:, -1, :])
        # out = self.fc1(torch.relu(out))
        return out

class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,dropout):
        super(Lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,dropout=dropout).cuda()
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()

    def forward(self, x):
        if(len(x.size()) == 2):
            x = torch.unsqueeze(x, 1)
        if (len(x.size()) == 1):
            x = torch.unsqueeze(torch.unsqueeze(x, 0), 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        out,(_,_) = self.lstm(x, (h0.detach(), c0.detach())) #out, (hn, cn)
        out = self.fc(out[:, -1, :])
        return out

    def convertTo3D(self,x):
        pass



class CnnGru(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(self).__init__()
        #TODO