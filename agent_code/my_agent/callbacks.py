import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # Extract features from the game state
    features = state_to_features(game_state)
    
    # Get the agent's position
    agent_x, agent_y = game_state['self'][-1]
    coins = game_state['coins']
    
    if coins:
        self.logger.debug("Navigating towards the nearest coin.")
        # Find the closest coin
        distances_to_coins = [abs(agent_x - x) + abs(agent_y - y) for x, y in coins]
        closest_coin_idx = np.argmin(distances_to_coins)
        coin_x, coin_y = coins[closest_coin_idx]

        # Determine the best move to get closer to the coin
        if coin_x > agent_x:
            return 'RIGHT'
        elif coin_x < agent_x:
            return 'LEFT'
        elif coin_y > agent_y:
            return 'DOWN'
        elif coin_y < agent_y:
            return 'UP'
    
    # No coins available or already at the coin's location, return 'WAIT'
    return 'WAIT'

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    # Get the agent's position
    agent_x, agent_y = game_state['self'][-1]

    # **1. Coins**
    # Calculate Manhattan distance to the nearest coin
    coins = game_state['coins']
    if coins:
        distances_to_coins = [abs(agent_x - x) + abs(agent_y - y) for x, y in coins]
        closest_coin_dist = min(distances_to_coins)
        features = [closest_coin_dist]  # Add the closest coin distance as a feature
        print(f"Agent Position: ({agent_x}, {agent_y}), Coins: {coins}, Closest Coin Distance: {closest_coin_dist}")
    else:
        features = [-1]  # No coins available
        print("No coins detected.")

    # Convert the list to a numpy array
    return np.array(features, dtype=float)
    """
    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
    """
