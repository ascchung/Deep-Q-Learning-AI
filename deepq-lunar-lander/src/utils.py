import random
import torch
import numpy as np


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        states = (
            torch.from_numpy(np.vstack([e[0] for e in experiences]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e[1] for e in experiences]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e[2] for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e[3] for e in experiences]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8))
            .float()
            .to(self.device)
        )
        return states, next_states, actions, rewards, dones

    def __len__(self):
        return len(self.memory)
