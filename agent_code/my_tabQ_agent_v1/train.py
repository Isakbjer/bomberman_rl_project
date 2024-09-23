from collections import deque
import pickle
from typing import List
import events as e
from .my_tabQ import MyRLAgent, ACTIONS

def setup_training(self):
    """
    Initialize training-related variables.
    """
    self.agent.transitions = deque(maxlen=3)

def game_events_occurred(self, old_game_state, self_action, new_game_state, events: List[str]):
    """
    Handle game events, store transitions in replay buffer, and perform Q-learning updates.
    """
    old_state = self.agent.state_to_features(old_game_state)
    new_state = self.agent.state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    self.agent.store_transition(old_state, self_action, reward, new_state)

    # Train from replay buffer
    self.agent.train_from_replay()

    self.agent.decay_epsilon()


def end_of_round(self, last_game_state, last_action, events: List[str]):
    """
    Called at the end of each game to finalize Q-learning updates and save the Q-table.
    """
    last_state = self.agent.state_to_features(last_game_state)
    reward = reward_from_events(self, events)

    self.agent.update_q_table(last_state, last_action, None, reward)
    self.agent.decay_epsilon()

    # Save the Q-table
    with open("q_table.pkl", "wb") as file:
        pickle.dump(self.agent.q_table, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Assign rewards to various events to encourage desired behaviors.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 7,
        e.KILLED_SELF: -25,
        e.INVALID_ACTION: -1,
        e.BOMB_DROPPED: 0.0,
        e.MOVED_LEFT: 0.01,
        e.MOVED_RIGHT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        e.WAITED: -0.1,
        e.CRATE_DESTROYED: 1,
        e.COIN_FOUND: 2,
        e.GOT_KILLED: -5,
        e.OPPONENT_ELIMINATED: 3,
        e.SURVIVED_ROUND: 1.5,
        'ESCAPED_DANGER': 2,
        'MOVE_CLOSER_TO_COIN': 0.5,
    }

    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    return reward_sum
