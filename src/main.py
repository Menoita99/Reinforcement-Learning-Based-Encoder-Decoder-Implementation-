import torch
from pathlib import Path
import datetime
from src.metricLogger import MetricLogger
from src.agent import Agent
from src.enviroment import Environment

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")


save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

agent = Agent(feature_dim=4, save_dir=save_dir)
env = Environment()

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
