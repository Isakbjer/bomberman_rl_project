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
        The field feature will be a 3x3 grid around the agent indicating walls, crates, and empty spaces.
        """
        if game_state is None:
            return None

        agent_x, agent_y = game_state['self'][-1]

        # 3x3 grid around the agent to represent walls (-1), crates (1), and empty spaces (0)
        field = game_state['field']
        local_grid = [
            [field[agent_x + dx, agent_y + dy] if 0 <= agent_x + dx < field.shape[0] and 0 <= agent_y + dy < field.shape[1] else -1
            for dx in [-1, 0, 1]]
            for dy in [-1, 0, 1]
        ]

        coins = game_state['coins']
        if coins:
            nearest_coin = min(coins, key=lambda c: abs(agent_x - c[0]) + abs(agent_y - c[1]))
            coin_distance = (nearest_coin[0] - agent_x, nearest_coin[1] - agent_y)
        else:
            coin_distance = (0, 0)

        bombs = game_state['bombs']
        bomb_distances = []
        danger = 0
        for (bx, by), _ in bombs:
            distance_to_bomb = abs(agent_x - bx) + abs(agent_y - by)
            bomb_distances.append(distance_to_bomb)
            if distance_to_bomb <= 3:
                danger = 1
        nearest_bomb_distance = min(bomb_distances) if bomb_distances else float('inf')

        bomb_available = int(game_state['self'][2])

        enemies = [game_state['others'][i][-1] for i in range(len(game_state['others']))]
        if enemies:
            nearest_enemy = min(enemies, key=lambda e: abs(agent_x - e[0]) + abs(agent_y - e[1]))
            enemy_distance = (nearest_enemy[0] - agent_x, nearest_enemy[1] - agent_y)
        else:
            enemy_distance = (0, 0)

        # Combine all features
        features = (
            coin_distance,            # Distance to the nearest coin
            nearest_bomb_distance,    # Distance to the nearest bomb
            danger,                   # Whether agent is in bomb danger zone
            bomb_available,           # Whether the agent has bombs available
            enemy_distance,           # Distance to the nearest enemy
            local_grid                # 3x3 grid representing the local field
        )

        # Canonicalize the state features by applying all symmetries
        return canonicalize_state(features)

    def act(self, game_state):
        """
        Choose an action based on a blend of Q-learning and rule-based actions.
        """
        state = self.state_to_features(game_state)
        agent_position = game_state['self'][-1]
        legal_moves = self.get_legal_moves(agent_position, game_state)
        
        if random.random() < self.epsilon:
            rb_action = rule_based.act(self.rb_agent, game_state)
            if rb_action in legal_moves:
                return rb_action
            else:
                return np.random.choice(legal_moves)
        if state in self.q_table:
            best_action_index = np.argmax(self.q_table[state])
            best_action = ACTIONS[best_action_index]
            if best_action in legal_moves:
                return best_action
            else:
                return np.random.choice(legal_moves)  
        else:
            return np.random.choice(legal_moves)

    def get_legal_moves(self, agent_position, game_state):
        """
        Returns the legal moves that the agent can take, considering walls and crates.
        """
        x, y = agent_position
        field = game_state['field']  # The game board with walls (-1), empty spaces (0), and crates (1)
        
        legal_moves = []
        
        if field[x, y - 1] == 0:  
            legal_moves.append('UP')
        if field[x, y + 1] == 0:  
            legal_moves.append('DOWN')
        if field[x - 1, y] == 0: 
            legal_moves.append('LEFT')
        if field[x + 1, y] == 0:
            legal_moves.append('RIGHT')
        
        # The WAIT action is always legal
        legal_moves.append('WAIT')
        
        if game_state['self'][2]:  # game_state['self'][2] is the bomb flag 
            legal_moves.append('BOMB')

        return legal_moves
    

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

def rotate_state_90(features):
    """
    Rotate the state by 90 degrees clockwise.
    For the feature vector, this will swap the distances accordingly. 
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    rotated_grid = np.rot90(local_grid, k=-1).tolist()  
    return ((-coin_y, coin_x), bomb_dist, danger, bomb_available, (-enemy_y, enemy_x), rotated_grid)


def rotate_state_180(features):
    """
    Rotate the state by 180 degrees (or twice by 90)
    """
    return rotate_state_90(rotate_state_90(features))

def rotate_state_270(features):
    """
    Rotate the state thrice by 90 degrees
    """
    return rotate_state_90(rotate_state_180(features))

def flip_state_vertical(features):
    """
    Flip the state vertically.
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    flipped_grid = np.flipud(local_grid).tolist()  # Flip grid vertically
    return ((coin_x, -coin_y), bomb_dist, danger, bomb_available, (enemy_x, -enemy_y), flipped_grid)

def flip_state_horizontal(features):
    """
    Flip the state horizontally.
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    flipped_grid = np.fliplr(local_grid).tolist()  # Flip grid horizontally
    return ((-coin_x, coin_y), bomb_dist, danger, bomb_available, (-enemy_x, enemy_y), flipped_grid)

def canonicalize_state(features):
    """
    Apply all possible rotations and flips and return the minimal state (canonical form).
    """
    transformations = [
        features,
        rotate_state_90(features),
        rotate_state_180(features),
        rotate_state_270(features),
        flip_state_horizontal(features),
        flip_state_vertical(features),
    ]
    
    # Return the lexicographically smallest representation
    return min(transformations)
