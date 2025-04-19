import gymnasium as gym
from src.config import Config


def train_agent():
    env = gym.make("LunarLander-v3")
    state_shape = env.observation_space.shape
    state_size = state_shape[0]
    number_actions = env.action_space.n

    print("State shape:", state_shape)
    print("State size:", state_size)
    print("Number of actions:", number_actions)

    # Rest of your training loop goes here...
