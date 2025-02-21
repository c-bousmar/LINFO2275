from BoardGame import BoardGame
from Enum import CellType, DieType, PositionType
import QLearningDecision
from markovDecision import markovDecision

import numpy as np

class DiceStrategy(BoardGame):
    
    def __init__(self, layout, circle):
        super().__init__(layout, circle)
        self.optimal_MDP_policy = markovDecision(layout, circle)[1]
        self.optimal_QLearning_policy = QLearningDecision(layout, circle)[1]
        self.strategies = {
            "Optimal_MDP" : self.optimal_MDP_strategy,
            "Always_Security" : self.always_choose_security,
            "Always_Normal" : self.always_choose_normal,
            "Always_Risky" : self.always_choose_risky,
            "Random" : self.random_strategy,
            "Optimal_QLearning" : self.optimal_QLearning_strategy,
            "Risky_Then_Cautious" : self.risky_then_cautious,
            "Greedy" : self.greedy_strategy,
        }
        
    def always_choose_security(self, position):
        return DieType.SECURITY.value

    def always_choose_normal(self, position):
        return DieType.NORMAL.value

    def always_choose_risky(self, position):
        return DieType.RISKY.value

    def random_strategy(self, position):
        return np.random.choice([DieType.SECURITY.value, DieType.NORMAL.value, DieType.RISKY.value], p=[1/3, 1/3, 1/3])

    def optimal_MDP_strategy(self, position):
        return self.optimal_MDP_policy[position]
    
    def optimal_QLearning_strategy(self, position):
        return self.optimal_QLearning_policy[position]

    def risky_then_cautious(self, position):
        if position < PositionType.SLOW_LANE_FIRST_CELL.value:
            return DieType.RISKY.value
        elif position < PositionType.FAST_LANE_FIRST_CELL.value:
            return DieType.NORMAL.value
        else:
            return DieType.SECURITY.value
        
    def greedy_strategy(self, position):
        best_die = DieType.RISKY.value
        
        for die in self.dice:
            for move, prob in zip(die.moves, die.probabilities):
                if prob > 0:
                    if not self.circle:
                        new_position = min(position + move, PositionType.FINAL_CELL.value)
                    else:
                        new_position = (position + move) % len(self.layout)
                    new_state = self.states[new_position]
                    
                    if new_state.cell_type in {CellType.PENALTY, CellType.RESTART, CellType.PRISON}:
                        return DieType.SECURITY.value
        
        return best_die