from torch import nn
import numpy as np
import pandas as pd

class Test(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(device="cuda")
        
        self.net = nn.Sequential()
        self.net.add_module('policy',self.policy)
        self.net.add_module('Relu',nn.ReLU())
        self.net.add_module('out layer',nn.Linear(output_dim, 1))


df = pd.read_csv(r'data\{}.csv'.format('BTC_USD'))
df = df.dropna(how='any', axis=0)
df = df[['Open','High','Low','Close']]
df = df.astype(np.float32)
print(df.dtypes)

"""
Row_list = []
for index, rows in df.iterrows():
    row = [rows.Open, rows.High, rows.Low, rows.Close]
    Row_list.append(row)

print(Row_list)
"""
