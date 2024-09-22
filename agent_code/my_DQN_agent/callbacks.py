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
    if game_state is None:
        return None

    # Get agent's position and basic game information
    agent_x, agent_y = game_state['self'][-1]
    field = game_state['field']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    bombs = game_state['bombs']
    enemies = game_state['others']

    # Initialize masks
    field_layout_mask = np.zeros_like(field)  
    coin_mask = np.zeros_like(field)  
    explosion_danger_mask = np.zeros_like(field) 
    future_explosion_mask = np.zeros_like(field)  
    bomb_mask = np.zeros_like(field) 
    enemy_position_mask = np.zeros_like(field) 
    escape_routes_mask = np.zeros_like(field) 

    # Field layout: Obstacles and crates
    field_layout_mask[np.where(field == -1)] = -1  # Walls
    field_layout_mask[np.where(field == 1)] = 1  # Crates

    for coin in coins:
        coin_mask[coin] = 1

    # Current explosion danger 
    explosion_danger_mask[np.where(explosion_map > 0)] = 1 

    # Future explosion danger (from bombs)
    for (bomb_x, bomb_y), countdown in bombs:
        # Bomb locations contribute to future danger
        future_explosion_mask[bomb_x, bomb_y] = 1
        
        # Mark the blast radius in all four directions (up, down, left, right)
        for (i, j) in [(bomb_x + h, bomb_y) for h in range(-3, 4)] + [(bomb_x, bomb_y + h) for h in range(-3, 4)]:
            if 0 <= i < field.shape[0] and 0 <= j < field.shape[1] and field[i, j] != -1:  # Don't mark past walls
                future_explosion_mask[i, j] = 1

    # Bombs: Positions and blast radius
    for (bomb_x, bomb_y), countdown in bombs:
        bomb_mask[bomb_x, bomb_y] = countdown
        for (i, j) in [(bomb_x + h, bomb_y) for h in range(-3, 4)] + [(bomb_x, bomb_y + h) for h in range(-3, 4)]:
            if 0 <= i < field.shape[0] and 0 <= j < field.shape[1]:
                bomb_mask[i, j] = min(bomb_mask[i, j], countdown)

    for enemy in enemies:
        enemy_position_mask[enemy[-1]] = 1

    # Escape routes: Identify free tiles for escape, want to maybe make this better, make it a concrete function later on?
    free_tiles = (field == 0) & (explosion_map == 0) & (bomb_mask == 0) & (future_explosion_mask == 0)
    escape_routes_mask[np.where(free_tiles)] = 1

    # **Valid Actions**: Determine possible moves
    valid_actions = [0, 0, 0, 0, 0, 0]  # [UP, RIGHT, DOWN, LEFT, WAIT, BOMB]
    if agent_y > 0 and free_tiles[agent_x, agent_y - 1]:
        valid_actions[0] = 1  # UP
    if agent_x < field.shape[0] - 1 and free_tiles[agent_x + 1, agent_y]:
        valid_actions[1] = 1  # RIGHT
    if agent_y < field.shape[1] - 1 and free_tiles[agent_x, agent_y + 1]:
        valid_actions[2] = 1  # DOWN
    if agent_x > 0 and free_tiles[agent_x - 1, agent_y]:
        valid_actions[3] = 1  # LEFT
    valid_actions[4] = 1  # WAIT
    if game_state['self'][2] and free_tiles[agent_x, agent_y]:
        valid_actions[5] = 1  # BOMB

    # Combine masks into a multi-channel array
    combined_features = np.stack([
        field_layout_mask,
        coin_mask,
        explosion_danger_mask, 
        future_explosion_mask,
        bomb_mask,
        enemy_position_mask,
        escape_routes_mask
    ])

    additional_features = np.array([
        int(game_state['self'][2]),  # Bomb availability
        1 if explosion_map[agent_x, agent_y] > 0 else 0,  # Immediate danger
        *valid_actions
    ], dtype=float)

    combined_features_flat = combined_features.flatten()
    features = np.concatenate([combined_features_flat, additional_features])

    return features