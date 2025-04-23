import gymnasium as gym
import torch
import imageio
import os

from src.dqn_agent import Agent
from src.config import Config


def load_agent(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    agent.local_qnetwork.load_state_dict(torch.load("checkpoint.pth"))
    return agent, env


def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state, epsilon=0.0)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave("video.mp4", frames, fps=30)
    print("Saved video to video.mp4")


def show_video():
    print("Opening video.mp4...")
    os.system("open video.mp4")


if __name__ == "__main__":
    agent, env = load_agent("LunarLander-v3")
    show_video_of_model(agent, "LunarLander-v3")
    show_video()
