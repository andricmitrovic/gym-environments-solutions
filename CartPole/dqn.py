import gym
import random
import os
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

    def save(self, filename='model.pth'):
        filename = os.path.join('./models', filename)
        torch.save(self.state_dict(), filename)


class Agent():
    def __init__(self, state_size, action_size, lr = 1e-4):
        self.model_path = "./models/model.pth"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.learning_rate = lr
        self.gamma = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.steps_done = 0

        self.policy_net = DeepQN(input_size=self.state_size, output_size=self.action_size).to(device)
        self.target_net = DeepQN(input_size=self.state_size, output_size=self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

    def load_model(self):
        """Loads the model weights from the path set in the object."""
        self.policy_net.load_state_dict(torch.load(self.model_path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state)).item()
        else:
            return random.randrange(self.action_size)

    def remember(self, state, action, reward, next_state, done):
        """Saves experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, sample_batch_size):
        """Learning from experience in memory."""

        if len(self.memory) < sample_batch_size:
            return  # Don't learn until we have at least batch size experiences

        sample_batch = random.sample(self.memory, sample_batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done = zip(*sample_batch)

        state_batch = torch.stack(state_batch, dim=0).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(dim=1).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
        next_state_batch = torch.stack(next_state_batch, dim=0).to(device)

        non_final_mask = ~torch.tensor(done, dtype=torch.bool)
        non_final_next_states = []
        for idx, next_state in enumerate(next_state_batch):
            if not done[idx]:
                non_final_next_states.append(next_state_batch[idx])
        non_final_next_states = torch.stack(non_final_next_states)

        # Predicted Q values with current state
        q_old = self.policy_net(state_batch).gather(1, action_batch)
        q_next = torch.zeros(sample_batch_size, device=device)
        with torch.no_grad():
            q_next[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_return = (q_next * self.gamma) + reward_batch

        # Compute loss
        loss = self.criterion(q_old, expected_return.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
