from Enum import DieType, PositionType, CellType
import QLearningDecision
from markovDecision import markovDecision

import numpy as np # type: ignore

optimal_MDP_policy = markovDecision.markovDecision(layout, circle)[1]
optimal_QLearning_policy = QLearningDecision(layout, circle)[1]

def always_choose_security(position, **kwargs):
    return DieType.SECURITY.value

def always_choose_normal(position, **kwargs):
    return DieType.NORMAL.value

def always_choose_risky(position, **kwargs):
    return DieType.RISKY.value

def random_strategy(position, **kwargs):
    return np.random.choice(
        [DieType.SECURITY.value, DieType.NORMAL.value, DieType.RISKY.value],
        p=[1/3, 1/3, 1/3]
    )

def optimal_MDP_strategy(position, board, **kwargs):
    return board.optimal_MDP_policy[position]

def optimal_QLearning_strategy(position, board, **kwargs):
    return board.optimal_QLearning_policy[position]

def risky_then_cautious(position, **kwargs):
    if position < PositionType.SLOW_LANE_FIRST_CELL.value:
        return DieType.RISKY.value
    elif position < PositionType.FAST_LANE_FIRST_CELL.value:
        return DieType.NORMAL.value
    else:
        return DieType.SECURITY.value

def greedy_strategy(position, board, **kwargs):
    best_die = DieType.RISKY.value
    for die in board.dice:
        for move, prob in zip(die.moves, die.probabilities):
            if prob > 0:
                if not board.circle:
                    new_position = min(position + move, PositionType.FINAL_CELL.value)
                else:
                    new_position = (position + move) % len(board.layout)
                new_state = board.states[new_position]
                if new_state.cell_type in {CellType.PENALTY, CellType.RESTART, CellType.PRISON}:
                    return DieType.SECURITY.value
    return best_die