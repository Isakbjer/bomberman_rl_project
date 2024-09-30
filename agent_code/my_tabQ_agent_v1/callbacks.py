import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    If in training mode, initialize a new Q-table, otherwise load an existing one.
    """
    global epsilon
    if os.path.isfile("q_table.pkl"):
        self.logger.info("Loading existing Q-table.")
        with open("q_table.pkl", "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.logger.info("Initializing new Q-table from scratch.")
        self.q_table = {}

    # Store epsilon values in self
    self.epsilon = 1.0         # Initial exploration rate
    self.epsilon_min = 0.1     # Minimum exploration rate
    self.epsilon_decay = 0.995 # Decay rate per round

    epsilon = 1.0  # Reset epsilon at the start of training

def decay_epsilon():
    """
    Decays the epsilon value after each round or step.
    """
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    epsilon = max(epsilon_min, epsilon)  # Ensure epsilon doesn't go below the minimum value


def get_legal_moves(agent_position, game_state):
    """
    Returns the legal moves that the agent can take, considering walls and crates.
    """
    x, y = agent_position
    field = game_state['field']  # The game board with walls (-1), empty spaces (0), and crates (1)
    
    legal_moves = []
    
    # Check if the moves UP, DOWN, LEFT, RIGHT are within bounds and don't hit walls/crates
    if field[x, y - 1] == 0:  # UP
        legal_moves.append('UP')
    if field[x, y + 1] == 0:  # DOWN
        legal_moves.append('DOWN')
    if field[x - 1, y] == 0:  # LEFT
        legal_moves.append('LEFT')
    if field[x + 1, y] == 0:  # RIGHT
        legal_moves.append('RIGHT')
    
    # The WAIT action is always legal
    legal_moves.append('WAIT')
    
    # The BOMB action is legal if the agent has a bomb available
    if game_state['self'][2]:  # game_state['self'][2] is the bomb availability flag
        legal_moves.append('BOMB')

    return legal_moves

def act(self, game_state: dict) -> str:
    """
    Choose an action based on the Q-table, restricted to legal moves.
    If training, it applies an epsilon-greedy policy to explore actions.
    """
    # Get the current state in features
    state = state_to_features(game_state)

    # Get the agent's position
    agent_position = game_state['self'][-1]

    # Get the list of legal moves
    legal_moves = get_legal_moves(agent_position, game_state)

    # Ensure the state is in the Q-table, otherwise initialize it
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))  # Initialize Q-values for all actions
    
    # Epsilon-greedy action selection for training mode
    if self.train:
        if random.random() < self.epsilon:
            self.logger.info(f"Exploring: Choosing random action from legal moves. Epsilon: {self.epsilon:.4f}")
            return np.random.choice(legal_moves)  # Choose a random legal move
        else:
            self.logger.info(f"Exploiting: Choosing best action from Q-table. Epsilon: {self.epsilon:.4f}")
            # Choose the best action from the Q-table, restricted to legal moves
            best_action_index = np.argmax([self.q_table[state][ACTIONS.index(move)] for move in legal_moves])
            best_action = legal_moves[best_action_index]
            self.logger.info(f"Best action selected: {best_action}")
            return best_action
    else:
        # Exploit the best action in testing mode, restricted to legal moves
        self.logger.info("Querying model for the best action during testing.")
        best_action_index = np.argmax([self.q_table[state][ACTIONS.index(move)] for move in legal_moves])
        best_action = legal_moves[best_action_index]
        self.logger.info(f"Best action selected: {best_action}")
        return best_action



def state_to_features(game_state: dict):
    """
    Convert the game state to a feature representation.
    The field feature will be a 3x3 grid around the agent indicating walls, crates, and empty spaces.
    It also applies symmetry transformations to reduce redundant Q-table entries.
    """
    if game_state is None:
        return None

    agent_x, agent_y = game_state['self'][-1]

    # 3x3 grid around the agent to represent walls (-1), crates (1), and empty spaces (0)
    field = game_state['field']
    local_grid = [
        [field[agent_x + dx, agent_y + dy] if 0 <= agent_x + dx < field.shape[0] and 0 <= agent_y + dy < field.shape[1] else -1
        for dy in [-1, 0, 1]]
        for dx in [-1, 0, 1]
    ]

    # Distance to the nearest coin
    coins = game_state['coins']
    if coins:
        nearest_coin = min(coins, key=lambda c: abs(agent_x - c[0]) + abs(agent_y - c[1]))
        coin_distance = (nearest_coin[0] - agent_x, nearest_coin[1] - agent_y)
    else:
        coin_distance = (0, 0)

    # Distance to the nearest bomb and danger indicator
    bombs = game_state['bombs']
    bomb_distances = []
    danger = 0
    for (bx, by), _ in bombs:
        distance_to_bomb = abs(agent_x - bx) + abs(agent_y - by)
        bomb_distances.append(distance_to_bomb)
        if distance_to_bomb <= 3:
            danger = 1
    nearest_bomb_distance = min(bomb_distances) if bomb_distances else float('inf')

    # Whether the agent has a bomb available
    bomb_available = int(game_state['self'][2])

    # Distance to the nearest enemy
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
        tuple(map(tuple, local_grid))  # 3x3 grid representing the local field as tuples
    )

    # Canonicalize the state features by applying symmetry transformations
    return canonicalize_state(features)

# Symmetry Reduction Functions

def rotate_state_90(features):
    """
    Rotate the state by 90 degrees clockwise.
    For the feature vector, this will swap the distances accordingly.
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    rotated_grid = np.rot90(local_grid, k=-1).tolist()  
    return ((-coin_y, coin_x), bomb_dist, danger, bomb_available, (-enemy_y, enemy_x), tuple(map(tuple, rotated_grid)))


def rotate_state_180(features):
    """
    Rotate the state by 180 degrees (or twice by 90).
    """
    return rotate_state_90(rotate_state_90(features))


def rotate_state_270(features):
    """
    Rotate the state thrice by 90 degrees.
    """
    return rotate_state_90(rotate_state_180(features))


def flip_state_vertical(features):
    """
    Flip the state vertically.
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    flipped_grid = np.flipud(local_grid).tolist()  # Flip grid vertically
    return ((coin_x, -coin_y), bomb_dist, danger, bomb_available, (enemy_x, -enemy_y), tuple(map(tuple, flipped_grid)))


def flip_state_horizontal(features):
    """
    Flip the state horizontally.
    """
    (coin_x, coin_y), bomb_dist, danger, bomb_available, (enemy_x, enemy_y), local_grid = features
    flipped_grid = np.fliplr(local_grid).tolist()  # Flip grid horizontally
    return ((-coin_x, coin_y), bomb_dist, danger, bomb_available, (-enemy_x, enemy_y), tuple(map(tuple, flipped_grid)))


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