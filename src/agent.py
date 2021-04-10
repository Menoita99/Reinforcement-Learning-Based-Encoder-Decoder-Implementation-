from src.enviroment import Environment
from src.enviroment import Actions
import numpy as np
import torch
from src.metricLogger import MetricLogger
import random, datetime, os, copy
from src.decoder import PolicyNet
from collections import deque


class Agent:

    def __init__(self, save_dir,feature_dim,batch_size=32):
        self.env = Environment()
        self.memory = deque(maxlen=100000)
        self.feature_dim = feature_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.policyNet = PolicyNet(self.feature_dim, len(Actions)).float()
        self.targetNet = copy.deepcopy(self.policyNet)

        # Q_target parameters are frozen.
        for p in self.targetNet.parameters():
            p.requires_grad = False

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.05

        self.save_every = 1e4  # no. of experiences between saving Policy Net

        self.curr_step = 0

        self.batch_size = batch_size
        self.gamma = .9
        self.optimizer = torch.optim.Adam(self.policyNet.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()#Huber loss

        self.burnin = 5e2  # min. experiences before training
        self.learn_every = 1  # no. of experiences between updates to Q_online (3) speed up train
        self.sync_every = 5e2  # no. of experiences between Q_target &



    """Given a state, choose an epsilon-greedy action"""
    def act(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action = Actions.random()
        # EXPLOIT
        else:
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            action_values = self.policyNet(state)
            action_idx = torch.argmax(action_values, axis=1).item()
            action = Actions(action_idx)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action



    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([int(action)]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([int(action)])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    """Sample experiences from memory"""
    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    """Update policy action value (Q) function with a batch of experiences"""
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_policy(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def td_estimate(self, state, action):
        #TODO add encoder code here
        current_Q = self.policyNet(state)[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        print("Current Q {}",current_Q)
        return current_Q

    #line 13 paper
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.policyNet(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.targetNet(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_policy(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.targetNet.load_state_dict(self.policyNet.state_dict())

    def save(self):
        save_path = (self.save_dir / f"policy_net_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.policyNet.state_dict(), exploration_rate=self.exploration_rate),save_path,)
        print(f"policyNet saved to {save_path} at step {self.curr_step}")



    def train(self,episodes):
        logger = MetricLogger(self.save_dir)

        for e in range(episodes):

            state = self.env.reset()

            #Trade the market!
            while True:

                # Run agent on the state
                action = self.act(state)

                # Agent performs action
                next_state, reward, done, info = self.env.step(action)

                # Remember
                self.cache(state, next_state, action, reward, done)

                # Learn
                q, loss = self.learn()

                # Logging
                logger.log_step(reward, loss, q)

                # Update state
                state = next_state

                # Check if end of game
                if done:
                    break

            logger.log_episode()

            if e % 20 == 0:
                logger.record(episode=e, epsilon=self.exploration_rate, step=self.curr_step)