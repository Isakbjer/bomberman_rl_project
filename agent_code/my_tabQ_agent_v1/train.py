from collections import deque, namedtuple
import pickle
from typing import List
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS, decay_epsilon 


# Transition tuple to store (s, a, r, s')
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters, want to test with different ones
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9  # Gamma
TRANSITION_HISTORY_SIZE = 3  

def setup_training(self):
    """
    Initialize training-related variables.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.logger.info("Training setup complete. Transition buffer initialized.")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    This is called every step to process events and update the agent.
    """
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    
    old_coin_distance = distance_to_nearest_coin(old_game_state)
    new_coin_distance = distance_to_nearest_coin(new_game_state)

    if new_coin_distance < old_coin_distance:
        events.append("MOVED_CLOSER_TO_COIN")

    # Reward calculation based on events
    reward = reward_from_events(self, events)

    # Store the transition
    self.transitions.append(Transition(old_state, self_action, new_state, reward))

    # Update Q-table
    update_q_table(self, old_state, self_action, reward, new_state)

    self.logger.info(f"Processed game events: {events}, updated Q-table.")

def distance_to_nearest_coin(game_state: dict) -> int:
    """
    Calculate the Manhattan distance to the nearest coin from the agent's position.
    """
    agent_x, agent_y = game_state['self'][-1]
    coins = game_state['coins']
    
    if not coins:
        return float('inf')  # No coins on the board

    # Return the Manhattan distance to the nearest coin
    return min(abs(agent_x - coin[0]) + abs(agent_y - coin[1]) for coin in coins)


def update_q_table(self, old_state, action, reward, new_state):
    """
    Update Q-table using the Q-learning update rule.
    """
    action_index = ACTIONS.index(action)

    # Ensure the state is initialized in the Q-table
    if old_state not in self.q_table:
        self.q_table[old_state] = np.zeros(len(ACTIONS))

    # Calculate the maximum Q-value for the next state
    if new_state is not None and new_state in self.q_table:
        future_rewards = np.max(self.q_table[new_state])
    else:
        future_rewards = 0

    # Q-learning update rule
    self.q_table[old_state][action_index] += LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * future_rewards - self.q_table[old_state][action_index]
    )

    self.logger.info(f"Q-table updated for state {old_state} and action {action}")


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Finalizes the round, updates the Q-table for the last state-action pair, and decays epsilon.
    """
    # Convert the last game state to features
    last_state = state_to_features(last_game_state)

    # Calculate the reward from events
    reward = reward_from_events(self, events)

    # Update the Q-table for the last action
    update_q_table(self, last_state, last_action, reward, None)

    # Decay epsilon after the round
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    self.logger.info(f"Epsilon decayed to: {self.epsilon:.4f}")

    # Save the Q-table
    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.q_table, file)

    self.logger.info("Q-table saved after the round.")



def reward_from_events(self, events: list) -> int:
    """
    Assign rewards to events and log key events that occur.
    """
    # added event to make the rewards less sparse
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 15,
        e.KILLED_SELF: -15,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.1,
        e.MOVED_LEFT: 0.01,
        e.MOVED_RIGHT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        e.WAITED: -0.1,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 2,
        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 15,
        e.SURVIVED_ROUND: 1.5,
        'MOVED_CLOSER_TO_COIN': 0.5  # Custom event reward
    }

    reward_sum = 0

    for event in events:
        # Log when specific events happen
        if event == e.COIN_COLLECTED:
            self.logger.info("Coin collected!")
        elif event == e.KILLED_OPPONENT:
            self.logger.info("Opponent killed!")
        elif event == e.KILLED_SELF:
            self.logger.info("Agent killed itself.")
        elif event == e.CRATE_DESTROYED:
            self.logger.info("Crate destroyed!")
        elif event == e.GOT_KILLED:
            self.logger.info("Agent died to a bomb!")

        # Assign rewards
        reward_sum += game_rewards.get(event, 0)

    # Log the total reward for this step
    self.logger.info(f"Total reward for this step: {reward_sum}")
    return reward_sum