import numpy as np
from .my_DQN import MyDQNAgent, ACTIONS

def setup(self):
    """
    Initialize the DQN agent.
    """
    input_dim = 4  # Number of features in the state vector
    output_dim = len(ACTIONS)
    self.agent = MyDQNAgent(input_dim=input_dim, output_dim=output_dim)

def act(self, game_state: dict):
    """
    Choose an action using the DQN agent.
    """
    state = state_to_features(game_state)
    return self.agent.act(state, train=self.train)

def state_to_features(game_state: dict) -> np.array:
    """
    Convert the game state into a feature vector.
    This example assumes a simple feature space with only 4 features:
    - Coin distance (x, y)
    - Danger (0 or 1)
    - Bomb availability (0 or 1)
    """
    if game_state is None:
        return None

    agent_x, agent_y = game_state['self'][-1]
    
    # Find the nearest coin
    coins = game_state['coins']
    if coins:
        nearest_coin = min(coins, key=lambda c: abs(agent_x - c[0]) + abs(agent_y - c[1]))
        coin_distance = (nearest_coin[0] - agent_x, nearest_coin[1] - agent_y)
    else:
        coin_distance = (0, 0)

    # Check for danger
    bombs = game_state['bombs']
    danger = int(any(abs(agent_x - bx) + abs(agent_y - by) <= 3 for (bx, by), _ in bombs))

    # Bomb availability
    bomb_available = int(game_state['self'][2])

    return np.array([*coin_distance, danger, bomb_available], dtype=np.float32)
