import os
import pickle
import random
import numpy as np
from collections import deque
from types import SimpleNamespace
import logging

try:
    from ..rule_based_agent import callbacks as rule_based
except ImportError:
    rule_based = None


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
        
        self.replay_buffer = deque(maxlen=1000)  # Replay buffer with a maximum size, started with 1000 
        self.batch_size = 32

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

        self.rb_agent = SimpleNamespace()
        self.rb_agent.logger = logging.getLogger("rule-based initial policy")
        self.rb_agent.train = False

        rule_based.setup(self.rb_agent)
    
    def store_transition(self, old_state, action, reward, new_state):
        """
        Store the transition in the replay buffer.
        """
        self.replay_buffer.append((old_state, action, reward, new_state))


    def state_to_features(self, game_state):
        """
        Convert the game state to a feature representation.
        """
        if game_state is None:
            return None
        
        agent_x, agent_y = game_state['self'][-1]
        
        coins = game_state['coins']
        if coins:
            nearest_coin = min(coins, key=lambda c: abs(agent_x - c[0]) + abs(agent_y - c[1]))
            coin_distance = (nearest_coin[0] - agent_x, nearest_coin[1] - agent_y)
        else:
            coin_distance = (0, 0)

        # Danger features (simplified)
        bombs = game_state['bombs']
        danger = int(any(abs(agent_x - bx) + abs(agent_y - by) <= 3 for (bx, by), _ in bombs))

        bomb_available = int(game_state['self'][2])

        # Return a simple tuple representing the state
        return (coin_distance, danger, bomb_available)

    def act(self, game_state):
        """
        Choose an action based on a blend of Q-learning and rule-based actions.
        """
        state = self.state_to_features(game_state)
        
        if random.random() < self.epsilon:
            rb_action = rule_based.act(self.rb_agent, game_state)
            if rb_action not in ACTIONS:
                rb_action = np.random.choice(ACTIONS)  # Fallback to random action if invalid, had an error with this
            # Learn from rule-based actions too
            self.update_q_table(state, rb_action, None, 0)  # Store this action in Q-table
            return rb_action

        if state in self.q_table:
            return ACTIONS[np.argmax(self.q_table[state])]
        else:
            return np.random.choice(ACTIONS)

# trying different act methods
#    def act(self, game_state):
#        """
#        Choose an action based on the current game state.
#        Blend between rule-based and Q-learning actions.
#        """
#        state = self.state_to_features(game_state)
#        
#        if random.random() < self.epsilon:
#            # Blend with rule-based agent's actions
#            rb_action = rule_based.act(self.rb_agent, game_state)
#            return rb_action
#        
#        if state in self.q_table:
#            return ACTIONS[np.argmax(self.q_table[state])]
#        else:
#            return np.random.choice(ACTIONS)

    def train_from_replay(self):
        """
        Sample a batch of transitions from the replay buffer and update Q-table.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Don't start training until we have enough samples

        batch = random.sample(self.replay_buffer, self.batch_size)
        for old_state, action, reward, new_state in batch:
            self.update_q_table(old_state, action, new_state, reward)


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
        self.q_table[old_state][action_index] += self.learning_rate * (
            reward + self.gamma * future_rewards - self.q_table[old_state][action_index]
        )
    def decay_epsilon(self):
        """
        Decay the exploration rate.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
