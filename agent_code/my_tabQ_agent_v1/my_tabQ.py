import os
import pickle
import random
import numpy as np
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class MyRLAgent:
    def __init__(self):
        self.q_table = {}
        self.transitions = None
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.gamma = 0.99

    def setup(self, training=True):
        """
        Initialize the agent. Load an existing Q-table if it exists.
        """
        if training or not os.path.isfile("q_table.pkl"):
            self.q_table = {}
        else:
            with open("q_table.pkl", "rb") as file:
                self.q_table = pickle.load(file)
        self.transitions = deque(maxlen=3)

    def state_to_features(self, game_state):
        """
        Convert the game state to a feature representation.
        """
        if game_state is None:
            return None
        
        agent_x, agent_y = game_state['self'][-1]
        
        # Example: nearest coin features
        coins = game_state['coins']
        if coins:
            nearest_coin = min(coins, key=lambda c: abs(agent_x - c[0]) + abs(agent_y - c[1]))
            coin_distance = (nearest_coin[0] - agent_x, nearest_coin[1] - agent_y)
        else:
            coin_distance = (0, 0)

        # Danger features (simplified)
        bombs = game_state['bombs']
        danger = int(any(abs(agent_x - bx) + abs(agent_y - by) <= 3 for (bx, by), _ in bombs))

        # Bomb availability
        bomb_available = int(game_state['self'][2])

        # Return a simple tuple representing the state
        return (coin_distance, danger, bomb_available)

    def act(self, game_state):
        """
        Choose an action based on the current game state.
        """
        state = self.state_to_features(game_state)
        if random.random() < self.epsilon:
            return np.random.choice(ACTIONS)
        
        if state in self.q_table:
            return ACTIONS[np.argmax(self.q_table[state])]
        else:
            return np.random.choice(ACTIONS)

    def update_q_table(self, old_state, action, new_state, reward):
        """
        Update the Q-table using the Q-learning update rule.
        """
        if old_state not in self.q_table:
            self.q_table[old_state] = np.zeros(len(ACTIONS))
        
        action_index = ACTIONS.index(action)
        if new_state is not None:
            future_rewards = np.max(self.q_table.get(new_state, np.zeros(len(ACTIONS))))
        else:
            future_rewards = 0

        # Q-learning update rule
        self.q_table[old_state][action_index] += self.learning_rate * (reward + self.gamma * future_rewards - self.q_table[old_state][action_index])

    def decay_epsilon(self):
        """
        Decay the exploration rate.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
