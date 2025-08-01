import random
import numpy as np
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Simple feedforward neural network for estimating Q-values."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        input_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    """Fixed-size buffer that stores experience tuples for training."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )


class DQNAgent:
    """DQN agent encapsulating policy network, target network, and training logic."""

    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 1e-3)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 500)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.memory_size)
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using an epsilon-greedy policy."""
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randrange(self.policy_net.model[-1].out_features)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def update_epsilon(self) -> None:
        """Decay epsilon after each training step."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_end, self.epsilon)

    def optimize_model(self) -> None:
        """Sample a batch from memory and update network weights."""
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions)

        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()

        # Update target network periodically
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
