from src.train import train_dqn
from src.dqn_agent import Agent
import gymnasium as gym

env = gym.make('LunarLander-v3')
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
agent = Agent(state_size, number_actions)

if __name__ == "__main__":
    train_dqn(agent, env)
