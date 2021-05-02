from torch import nn
import numpy as np
import pandas as pd
import torch

class Test(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(  # dim 4 x 4
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding=1),  # output: 8 x 4 x 4
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1),  # output: 16 x 4 x 4
            nn.ReLU(),
      #      nn.MaxPool2d(2, 2),  # output: 16 x 2 x 2

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1),  # output: 32 x 4 x 4
            nn.ReLU(),
      #      nn.MaxPool2d(2, 2),  # output: 32 x 1 x 1

            nn.Flatten(),  # output batch_size x 128 , 128 was obtain experimentally
                           # without MaxPool = 1568
        )

    def forward(self, input):
        if len(input.size()) == 3:
            input = input.unsqueeze(1)
        elif len(input.size()) == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        return self.net(input)


test = Test()
state = [[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.]]
input = torch.tensor(state)
print(test(input).size())

"""
df = pd.read_csv(r'data\{}.csv'.format('BTC_USD'))
df = df.dropna(how='any', axis=0)
df = df[['Open','High','Low','Close']]
df = df.astype(np.float32)
print(df.dtypes)

Row_list = []
for index, rows in df.iterrows():
    row = [rows.Open, rows.High, rows.Low, rows.Close]
    Row_list.append(row)

print(Row_list)
"""
