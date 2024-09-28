import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MyDQNAgent:
    def __init__(self, input_dim, output_dim, training, learning_rate=0.001):
        self.q_network = DQNetwork(input_dim, output_dim)
        if not training:
            with open("my_DQN_model.pt", "rb") as f:
                self.q_network.load_state_dict(torch.load(f))
        self.target_network = DQNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        # Synchronize target network with the Q network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state, valid_moves, train=True):
        if train and random.random() < self.epsilon:
            return ACTIONS[np.random.choice(valid_moves)]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return ACTIONS[torch.argmax(q_values[:, valid_moves]).item()]

    def remember(self, state, action, next_state, reward):
        action_idx = ACTIONS.index(action)
        self.replay_buffer.append((state, action_idx, next_state, reward))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, next_states, rewards = zip(*batch)

        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array([ns for ns in next_states if ns is not None]), dtype=torch.float32)

        # Q-values for current states
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Target Q-values for next states
        next_q_values = torch.zeros(self.batch_size)
        if len(next_states_tensor) > 0:
            next_q_values[[ns is not None for ns in next_states]] = self.target_network(next_states_tensor).max(1)[0].detach()

        target_q_values = rewards_tensor + self.gamma * next_q_values

        # Update the model
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Synchronize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

