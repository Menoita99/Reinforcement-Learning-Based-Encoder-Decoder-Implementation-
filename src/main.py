from pathlib import Path
import datetime
import pandas as pd
import numpy as np
from src.agent import Agent
from src.encoder import Mlp, Cnn, Lstm
import torch
import matplotlib.pyplot as plt

from src.enviroment import Actions

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") / "Lstm(5,12,2,10,0.2)"
save_dir.mkdir(parents=True)
load_dir = Path("checkpoints") / "2021-07-17T07-33-08" / "Lstm(5,12,2,10,0.2)" / "policy_net_30.chkpt"
epochs = int(5e6)

def evaluate(agent):
    state = agent.env.reset()
    data = []

    while True:
        state = torch.tensor(state).cuda()
        action = torch.argmax(agent.net(state,model="policy"), axis=1)
        state, reward, done, info = agent.env.step(action)
        data.append([action,info[0],state[3]])
        if done:
            break

    prices = []
    buys = []
    sells = []
    bigmoney = []
    index = 0
    for action,money,price in data:

        print("price {} action {} money {}".format(price,str(Actions(action.item())),money))#price[3]
        prices.append(price)#price[3]
        if Actions(action.item()) == Actions.Buy:
            buys.append([index,price])#price[3]
        if Actions(action.item()) == Actions.Sell:
            sells.append([index,price])#price[3]
        if(price == 50539.00390625):
            buys.append([index, price])  # price[3]
        bigmoney.append(money)
        index += 1

    prices = np.array(prices)
    buys = np.array(buys)
    sells = np.array(sells)
    bigmoney = np.array(bigmoney)

    plt.figure(0)
    plt.plot(np.arange(len(bigmoney)),bigmoney)
    plt.plot(np.arange(len(prices)),prices)
    plt.savefig("results.jpg")
    plt.clf()

    plt.figure(1)
    plt.scatter(buys[:, 0], buys[:, 1], c='green',s=1)
    plt.scatter(sells[:, 0], sells[:, 1], c='red',s=1)
    plt.savefig("results_actions.jpg")
    plt.clf()


#agent = Agent(encoder=Cnn(),feature_dim=1568, hidden_dim=64,windowSize=4,useWindowState=True,save_dir=save_dir,seed=1,market="BTC_USD")
#agent = Agent(encoder=None, feature_dim=4, hidden_dim=8,save_dir=save_dir,seed=1)

# #
agent = Agent(encoder=Cnn(1,16),useWindowState=True,windowSize=10, feature_dim=16, hidden_dim=24,save_dir=save_dir,seed=453)
agent.train(epochs)
evaluate(agent)



