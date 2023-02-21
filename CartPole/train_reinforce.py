import torch
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


def train_reinforce(alfa):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PolicyNet(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear1 = nn.Linear(input_size, 64)
            self.linear2 = nn.Linear(64, output_size)
            self.relu = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.softmax(self.linear2(x))
            return x

    max_episodes = 1000
    lr = alfa
    γ = 0.99

    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = PolicyNet(state_size, action_size)
    optim = torch.optim.Adam(policy_net.parameters(), lr=lr)

    scores = []

    for _ in range(max_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float)
        done = False
        Actions, States, Rewards = [], [], []

        episode_score = 0
        while not done:
            # Get action
            probs = policy_net(state)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()

            # Act
            next_state, reward, done, _ = env.step(action)

            # Save trajectory
            Actions.append(torch.tensor(action, dtype=torch.int))
            States.append(state)
            Rewards.append(reward)

            state = torch.tensor(next_state, dtype=torch.float)

            episode_score += reward
        scores.append(episode_score)

        # # Early stopping
        # if index_episode > 30:
        #     last_30 = scores[-30:]
        #     avg = sum(last_30) / 30
        #     if avg > 480:
        #         # Save model weights
        #         # self.agent.model.save()
        #         # return True, scores, avg_score, agent.loss_history
        #         break

        # Calculate returns
        DiscountedReturns = []
        for t in range(len(Rewards)):
            G = 0.0
            for k, r in enumerate(Rewards[t:]):
                G += (γ ** k) * r
            DiscountedReturns.append(G)

        # Optimize network
        for S, A, G in zip(States, Actions, DiscountedReturns):
            probs = policy_net(S)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(A)

            loss = - log_prob * G

            optim.zero_grad()
            loss.backward()
            optim.step()

    return scores


if __name__ == '__main__':
    num_runs = 10
    alfas = [1e-3, 1e-4, 5e-5]
    scores = {x:list() for x in alfas}
    for run in tqdm(range(num_runs)):
        for alfa in alfas:
            scores[alfa].append(train_reinforce(alfa))

    for alfa in alfas:
        scores[alfa] = np.array(scores[alfa])
        scores[alfa] = np.mean(scores[alfa], axis = 0)
        plt.plot(range(len(scores[alfa])), scores[alfa], label = f'lr = {alfa}')
    plt.xlabel('EPISODES')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('scores_reinforce.png')
    plt.show()
