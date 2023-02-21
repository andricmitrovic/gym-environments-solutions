import gym
from dqn import Agent
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)
agent.load_model()
agent.policy_net.eval()

state = env.reset()

done = False
episode_score = 0
while not done:
    env.render()

    state = torch.tensor(state, dtype=torch.float32, device=device)
    # Choose action
    action = torch.argmax(agent.policy_net(state)).item()
    # Act
    next_state, reward, done, _ = env.step(action)

    state = next_state
    episode_score += reward

print(episode_score)
env.close()