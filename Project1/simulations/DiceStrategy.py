from BoardGame import BoardGame

from markovDecision import markovDecision
import QLearningDecision # type: ignore

import numpy as np # type: ignore

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
        return self.board.dice_types.SECURITY.value

    def always_choose_normal(self, position):
        return self.board.dice_types.NORMAL.value

    def always_choose_risky(self, position):
        return self.board.dice_types.RISKY.value

    def random_strategy(self, position):
        return np.random.choice([self.board.dice_types.SECURITY.value, self.board.dice_types.NORMAL.value, self.board.dice_types.RISKY.value],
                                p=[1/3, 1/3, 1/3])

    def optimal_MDP_strategy(self, position):
        return self.optimal_MDP_policy[position]
    
    def optimal_QLearning_strategy(self, position):
        return self.optimal_QLearning_policy[position]

    def risky_then_cautious(self, position):
        if position < self.board.positions_types.SLOW_LANE_FIRST_CELL.value:
            return self.board.dice_types.RISKY.value
        elif position < self.board.positions_types.FAST_LANE_FIRST_CELL.value:
            return self.board.dice_types.NORMAL.value
        else:
            return self.board.dice_types.SECURITY.value
        
    def greedy_strategy(self, position):
        best_die = self.board.dice_types.RISKY.value
        
        for die in self.dice:
            for move, prob in zip(die.moves, die.probabilities):
                if prob > 0:
                    if not self.circle:
                        new_position = min(position + move, self.board.positions_types.FINAL_CELL.value)
                    else:
                        new_position = (position + move) % len(self.layout)
                    new_state = self.states[new_position]
                    
                    if new_state.cell_type in {self.board.cells_types.PENALTY, self.board.cells_types.RESTART, self.board.cells_types.PRISON}:
                        return self.board.dice_types.SECURITY.value
        
        return best_die