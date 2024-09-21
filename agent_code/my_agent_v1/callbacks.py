import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
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
    if game_state is None:
        return 'WAIT'

    # Extract features from the game state
    features = state_to_features(game_state)

    # Retrieve valid actions from features
    valid_actions = features[-6:]  # The last 6 elements in the feature vector are [UP, RIGHT, DOWN, LEFT, WAIT, BOMB]

    # Map indices to actions
    action_mapping = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

    # Choose an action: For now, we choose the first valid action (this logic can be extended)
    for i, valid in enumerate(valid_actions):
        if valid:
            return action_mapping[i]

    # If no valid action found (failsafe), wait
    return 'WAIT'


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
    field_layout_mask = np.zeros_like(field)  # Mask 1: Obstacles and crates
    coin_mask = np.zeros_like(field)  # Mask 2: Coin positions
    explosion_danger_mask = np.zeros_like(field)  # Mask 3: Current explosion danger
    future_explosion_mask = np.zeros_like(field)  # Mask 4: Future explosion danger
    bomb_mask = np.zeros_like(field)  # Mask 5: Bombs and blast radii
    enemy_position_mask = np.zeros_like(field)  # Mask 6: Enemy positions
    escape_routes_mask = np.zeros_like(field)  # Mask 7: Safe tiles to escape to

    # Field layout: Obstacles and crates
    field_layout_mask[np.where(field == -1)] = -1  # Walls
    field_layout_mask[np.where(field == 1)] = 1  # Crates

    # Coins
    for coin in coins:
        coin_mask[coin] = 1

    # Current explosion danger
    explosion_danger_mask[np.where(explosion_map > 0)] = 1

    # Future explosion danger (from bombs)
    for (bomb_x, bomb_y), countdown in bombs:
        future_explosion_mask[bomb_x, bomb_y] = countdown
        
        # Mark the blast radius in all four directions (up, down, left, right)
        for (i, j) in [(bomb_x + h, bomb_y) for h in range(-3, 4)] + [(bomb_x, bomb_y + h) for h in range(-3, 4)]:
            if 0 <= i < field.shape[0] and 0 <= j < field.shape[1]:
                # Only update future explosion mask if the tile is free or contains a crate
                if field[i, j] == 0 or field[i, j] == 1:
                    future_explosion_mask[i, j] = min(future_explosion_mask[i, j], countdown) if future_explosion_mask[i, j] != 0 else countdown

    # Bombs: Positions and blast radius
    for (bomb_x, bomb_y), countdown in bombs:
        bomb_mask[bomb_x, bomb_y] = countdown  # Mark bomb position
        # Mark blast radius
        for (i, j) in [(bomb_x + h, bomb_y) for h in range(-3, 4)] + [(bomb_x, bomb_y + h) for h in range(-3, 4)]:
            if 0 <= i < field.shape[0] and 0 <= j < field.shape[1]:
                bomb_mask[i, j] = min(bomb_mask[i, j], countdown)

    # Enemies
    for enemy in enemies:
        enemy_position_mask[enemy[-1]] = 1

    # Escape routes: Identify free tiles for escape (exclude obstacles, crates, explosions, bombs, and future explosions)
    free_tiles = (field == 0) & (explosion_map == 0) & (bomb_mask == 0) & (future_explosion_mask == 0)
    escape_routes_mask[np.where(free_tiles)] = 1

    # **Valid Actions**: Determine possible moves based on free tiles, obstacles, explosions, bombs, and future explosions
    valid_actions = [0, 0, 0, 0, 0, 0]  # [UP, RIGHT, DOWN, LEFT, WAIT, BOMB]
    if agent_y > 0 and free_tiles[agent_x, agent_y - 1]:
        valid_actions[0] = 1  # UP
    if agent_x < field.shape[0] - 1 and free_tiles[agent_x + 1, agent_y]:
        valid_actions[1] = 1  # RIGHT
    if agent_y < field.shape[1] - 1 and free_tiles[agent_x, agent_y + 1]:
        valid_actions[2] = 1  # DOWN
    if agent_x > 0 and free_tiles[agent_x - 1, agent_y]:
        valid_actions[3] = 1  # LEFT

    # Always consider 'WAIT' as valid
    valid_actions[4] = 1  # WAIT

    # Check if placing a bomb is valid
    if game_state['self'][2] and free_tiles[agent_x, agent_y]:
        valid_actions[5] = 1  # BOMB

    # **Additional Features**: Bomb availability, immediate danger, etc.
    can_place_bomb = int(game_state['self'][2])
    immediate_danger = 1 if explosion_map[agent_x, agent_y] > 0 else 0

    # Combine masks into a multi-channel array
    combined_features = np.stack([
        field_layout_mask,
        coin_mask,
        explosion_danger_mask,
        future_explosion_mask,  # New future explosion mask
        bomb_mask,
        enemy_position_mask,
        escape_routes_mask
    ])

    # Additional features as a flat array
    additional_features = np.array([
        can_place_bomb,
        immediate_danger,
        *valid_actions  # Include valid actions in the state features
    ], dtype=float)

    # Flatten masks and concatenate with additional features
    combined_features_flat = combined_features.flatten()
    features = np.concatenate([combined_features_flat, additional_features])

    return features
