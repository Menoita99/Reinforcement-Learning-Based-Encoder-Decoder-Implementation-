from pathlib import Path
from datetime import datetime

from src.agent import Agent
from src.encoder import Mlp, Cnn
import torch

from src.enviroment import Actions

save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S") / "Cnn BTC_USD 128-64"
save_dir.mkdir(parents=True)
load_dir = Path("checkpoints") / "2021-04-27T00-35-01" / "Cnn BTC_USD 128-64" / "policy_net_38.chkpt"
epochs =    int(10e3)
#agent = Agent(encoder=Mlp(4,10),feature_dim=10, hidden_dim=20,save_dir=save_dir,seed=1,market="GOOGL")#,loadModelPath=load_dir)
agent = Agent(encoder=Cnn(),feature_dim=128, hidden_dim=64,windowSize=4,useWindowState=True,save_dir=save_dir,seed=1,market="BTC_USD")
#agent = Agent(encoder=None, feature_dim=4, hidden_dim=8,save_dir=save_dir,seed=1)
agent.train(epochs)

def evaluate():
    agent = Agent(encoder=Cnn(),feature_dim=128, hidden_dim=64,windowSize=4,useWindowState=True,save_dir=save_dir,seed=1,market="BTC_USD",loadModelPath=load_dir)

    state = agent.env.reset()
    data = []

    while True:
        state = torch.tensor(state).cuda()
        action = torch.argmax(agent.net(state,model="policy"), axis=1)
        state, reward, done, info = agent.env.step(action)
        data.append([action,info[0],state[3]])
        if done:
            break

    for action,money,price in data:
        print("price {} action {} money {}".format(price[3],str(Actions(action.item())),money))

