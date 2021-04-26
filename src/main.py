from pathlib import Path
from datetime import datetime

from src.agent import Agent
from src.encoder import Mlp, Cnn

save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S") / "MLP 4-10"
save_dir.mkdir(parents=True)
load_dir = Path("checkpoints") / "2021-04-25T03-58-09" / "Mlp 4-10_10-20" / "policy_net_24.chkpt"
epochs = int(15e3)
agent = Agent(encoder=Mlp(4,10),feature_dim=10, hidden_dim=20,save_dir=save_dir,seed=1,loadModelPath=None, market= "BTC_USD")
#agent = Agent(encoder=Cnn(),feature_dim=128, hidden_dim=64,windowSize=4,useWindowState=True,save_dir=save_dir,seed=1)
#agent = Agent(encoder=None, feature_dim=4, hidden_dim=8,save_dir=save_dir,seed=1)
agent.train(epochs)#int(5e4))

