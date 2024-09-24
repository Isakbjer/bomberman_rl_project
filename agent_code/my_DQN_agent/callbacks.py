import numpy as np
from collections import deque
from .my_DQN import MyDQNAgent, ACTIONS

from types import SimpleNamespace
import logging

try:
    from ..rule_based_agent import callbacks as rule_based
except ImportError:
    rule_based = None

INPUT_DIM = 29

ENEMY = 2

UP, RIGHT, DOWN, LEFT = range(4)
DIRECTIONS = {
    (0, -1): UP,
    (1, 0): RIGHT,
    (0, 1): DOWN,
    (-1, 0): LEFT,
}

def valid_moves(field, agent_x, agent_y, bomb, enemy_pos):
    ret = [4]  # WAIT
    
    for dir, idx in DIRECTIONS.items():
        tile = (agent_x + dir[0], agent_y + dir[1])
        if field[*tile] != 0:
            continue
        if tile in enemy_pos:
            continue
        ret.append(idx)
    
    if bomb:
        ret.append(5)

    return ret


def bfs(start, field, cb):
    q = deque([(start, None, 0)])
    seen = {start}

    while q:
        current, dir_from_start, dist = q.popleft()
        for dir in DIRECTIONS.keys():
            neigh = (current[0] + dir[0], current[1] + dir[1])
            if neigh in seen:
                continue

            if neigh[0] < 0 or neigh[1] < 0:
                continue
            if neigh[0] >= field.shape[0] or neigh[1] >= field.shape[1]:
                continue
            if field[*neigh] == -1:  # wall
                continue

            seen.add(neigh)

            cur_dir_from_start = dir if dir_from_start is None else dir_from_start
            if cb(neigh, cur_dir_from_start, dist + 1):
                return

            if field[*neigh] == 0:
                q.append((neigh, cur_dir_from_start, dist + 1))


def setup(self):
    """
    Initialize the DQN agent.
    """
    input_dim = INPUT_DIM # Number of features in the state vector
    output_dim = len(ACTIONS)
    self.agent = MyDQNAgent(input_dim=input_dim, output_dim=output_dim, training=self.train)
    if self.train:
        self.rb_agent = SimpleNamespace()
        self.rb_agent.logger = logging.getLogger("rule_based")
        self.rb_agent.train = False
        rule_based.setup(self.rb_agent)

def act(self, game_state: dict):
    """
    Choose an action using the DQN agent.
    """
    if self.train and len(self.agent.replay_buffer) < 10000:
        return rule_based.act(self.rb_agent, game_state)

    state, valid_moves = state_to_features(game_state)
    return self.agent.act(state, valid_moves, train=self.train)

def get_bomb_arm(field, bombs, agent_x, agent_y):
    for (bomb_x, bomb_y), timer in bombs:
        if agent_x == bomb_x and abs(agent_y - bomb_y) <= 3:
            lo, hi = min(agent_y, bomb_y), max(agent_y, bomb_y)
            if not(any(field[agent_x, lo:hi+1] == 1)):
                if agent_y == bomb_y: # center
                    return [0.5, 0.5, 0.5, 0.5], timer
                elif agent_y > bomb_y: # up arm
                    return [1, 0.5, 0, 0.5], timer
                else: # down arm
                    return [0, 0.5, 1, 0.5], timer
        elif agent_y == bomb_y and abs(agent_x - bomb_x) <= 3:
            lo, hi = min(agent_x, bomb_x), max(agent_x, bomb_x)
            if not(any(field[lo:hi+1, agent_y] == 1)):
                if agent_x == bomb_x:
                    return [0.5, 0.5, 0.5, 0.5], timer
                elif agent_x > bomb_x:
                    return [0.5, 1, 0.5, 0], timer
                else:
                    return [0.5, 0, 0.5, 1], timer
    return [0.5, 0.5, 0.5, 0.5], -1


# def bounds_check(field, coords):
#     if coords[0] < 0 or coords[1] < 0:
#         return False
#     if coords[0] >= field.shape[0] or coords[1] >= field.shape[1]:
#         return False
#     return True


def check_own_bomb_trap(field, agent_x, agent_y):
    d = 5

    for dir in DIRECTIONS.keys():
        for i in range(1, 3+1):
            coords = [agent_x + dir[0] * i, agent_y + dir[1] * i]

            # if not bounds_check(field, coords):
            #     break
            if field[*coords] != 0:
                break

            axis = 1 - dir.index(0)
            other_axis = 1 - axis

            for j in (-1, 1):
                coords2 = [*coords]
                coords2[other_axis] += j

                # if not bounds_check(field, coords2):
                #     continue
                if field[*coords2] == 0:
                    d = min(d, i + 1)
            
            if i == 3:
                coords2 = [*coords]
                coords2[axis] += dir[axis]

                # if bounds_check(field, coords2) and field[*coords2] == 0:
                if field[*coords2] == 0:
                    d = min(d, i + 1)
    
    return d == 5

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
    enemy_pos = {x[-1] for x in enemies}
    bombdict = {(x, y): t for ((x, y), t) in bombs}

    coin_dir, crate_dir, enemy_dir = [None]*3

    def obj_cb(pos, dir_from_start, dist):
        nonlocal coin_dir, crate_dir, enemy_dir

        feat = [0, 0, 0, 0]
        feat[DIRECTIONS[dir_from_start]] = 1

        if field[*pos] == 1 and not crate_dir:
            crate_dir = feat
        elif pos in coins and not coin_dir:
            coin_dir = feat
        elif pos in enemy_pos and not enemy_dir:
            enemy_dir = feat
        
        return crate_dir and coin_dir and enemy_dir
    
    # Find objects
    bfs((agent_x, agent_y), field, obj_cb)
    
    coin_dir = coin_dir or [0, 0, 0, 0]
    crate_dir = crate_dir or [0, 0, 0, 0]
    enemy_dir = enemy_dir or [0, 0, 0, 0]

    bomb_arm, timer = get_bomb_arm(field, bombs, agent_x, agent_y)
    own_bomb_trap = int(check_own_bomb_trap(field, agent_x, agent_y))

    crates_in_blast_radius = [0, 0, 0, 0]
    enemies_in_blast_radius = [0, 0, 0, 0]

    for dir, dir_ind in DIRECTIONS.items():
        cur_x, cur_y = agent_x, agent_y
        for i in range(3):
            cur_x += dir[0]
            cur_y += dir[1]
            # if not bounds_check(field, (cur_x, cur_y)):
            #     break
            if field[cur_x, cur_y] == -1:
                break
            elif field[cur_x, cur_y] == 1 and not crates_in_blast_radius[dir_ind]:
                crates_in_blast_radius[dir_ind] = (3 - i)/3
            elif (cur_x, cur_y) in enemy_pos and not enemies_in_blast_radius[dir_ind]:
                enemies_in_blast_radius[dir_ind] = (3 - i)/3
    
    movement_traps = [0, 0, 0, 0]
    for dir, dir_ind in DIRECTIONS.items():
        cur_x, cur_y = agent_x + dir[0], agent_y + dir[1]
        # if not bounds_check(field, (cur_x, cur_y)):
        #     continue
        if bombdict.get((cur_x, cur_y), 5) == 0 or explosion_map[cur_x, cur_y] > 1:
            movement_traps[dir_ind] = 1
    
    return np.concatenate([
        coin_dir,
        crate_dir,
        enemy_dir,
        bomb_arm,
        #[timer],
        [own_bomb_trap],
        crates_in_blast_radius,
        enemies_in_blast_radius,
        movement_traps
    ]), valid_moves(field, agent_x, agent_y, game_state['self'][-2], enemy_pos)

