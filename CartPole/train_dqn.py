import gym
import time
from dqn import Agent
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np


def train_dqn(lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CartPole-v1')
    EPISODES = 500
    BATCH_SIZE = 128
    TAU = 0.005

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, lr)

    scores = []
    for index_episode in range(EPISODES):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)

        done = False
        episode_score = 0
        while not done:
            # Choose action
            action = agent.act(state)

            # Act
            next_state, reward, done, _ = env.step(action)

            # Remember experience
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            episode_score += reward

            # Perform optimization
            agent.learn(BATCH_SIZE)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

        # Console output of learning process
        # print(f'Episode {index_episode + 1}/{EPISODES} Score: {episode_score}')

        # Save cumulative reward in this episode and an avarage reward until now
        scores.append(episode_score)

        # # Early stopping
        # if index_episode > 100:
        #     last_100 = scores[-100:]
        #     avg = sum(last_100) / 100
        #     if avg > 485:
        #         break

    # Save model weights
    # agent.policy_net.save()

    return scores


if __name__ == '__main__':

    num_runs = 10
    alfas = [1e-3, 1e-4, 5e-5]
    scores = {x:list() for x in alfas}
    for run in tqdm(range(num_runs)):
        for alfa in alfas:
            scores[alfa].append(train_dqn(alfa))

    for alfa in alfas:
        scores[alfa] = np.array(scores[alfa])
        scores[alfa] = np.mean(scores[alfa], axis = 0)
        plt.plot(range(len(scores[alfa])), scores[alfa], label = f'lr = {alfa}')
    plt.xlabel('EPISODES')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('scores_dqn.png')
    plt.show()

