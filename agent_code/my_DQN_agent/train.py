import torch
import pickle
from collections import deque
from .my_DQN import MyDQNAgent, ACTIONS
import events as e
from .callbacks import state_to_features, INPUT_DIM

def setup_training(self):
    self.transitions = deque(maxlen=3)
    # self.agent = MyDQNAgent(input_dim=INPUT_DIM, output_dim=len(ACTIONS), training=True)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    old_state, _ = state_to_features(old_game_state)
    new_state, _ = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    self.agent.remember(old_state, self_action, new_state, reward)
    self.agent.replay()

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    self.logger.debug(f"End of round with events: {events} last action {last_action}")
    self.agent.remember(state_to_features(last_game_state)[0], last_action, None, reward_from_events(self, events))

    # Save the model
    with open("my_DQN_model.pt", "wb") as file:
        torch.save(self.agent.q_network.state_dict(), file)

def reward_from_events(self, events: list) -> int:
    """
    Assign rewards for different events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_SELF: -25,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.1,
        e.MOVED_LEFT: 0.01,
        e.MOVED_RIGHT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        e.WAITED: -0.1
    }
    return sum(game_rewards.get(event, 0) for event in events)
