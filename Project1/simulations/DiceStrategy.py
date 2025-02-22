import sys
sys.path.append('/Users/mathisdelsart/Desktop/Github-Repository/LINFO2275/Project1')

from algorithms.MarkovDecision import markovDecision
from algorithms.QLearningDecision import QLearningDecision
from game.TransitionManager import TransitionManager
import numpy as np # type: ignore


class DiceStrategy:
    
    def __init__(self, board, strategy_names=None):
        self.board = board
        self.tm = TransitionManager(board)
        self.optimal_MDP_policy = None
        self.optimal_QLearning_policy = None
        self.possible_strategies = {
            "Optimal_MDP" : self.optimal_MDP_strategy,
            "Optimal_QLearning" : self.optimal_QLearning_strategy,
            "Always_Security" : self.always_choose_security,
            "Always_Normal" : self.always_choose_normal,
            "Always_Risky" : self.always_choose_risky,
            "Random" : self.random_strategy,
            "Risky_Then_Cautious" : self.risky_then_cautious
        }
        
        if strategy_names == None:
            self.strategies = self.possible_strategies
        else:
            self.strategies = {}
            for strategy_name in strategy_names:
                if strategy_name in self.possible_strategies:
                    if strategy_name == "Optimal_MDP": self.optimal_MDP_policy = markovDecision(self.board.layout, self.board.circle)[1]
                    if strategy_name == "Optimal_QLearning": self.optimal_QLearning_policy = QLearningDecision(self.board.layout, self.board.circle)[1]
                    self.strategies[strategy_name] = self.possible_strategies[strategy_name]
        
    def always_choose_security(self, position):
        for die in self.board.dice:
            if die.type == self.board.dice_types.SECURITY:
                return die

    def always_choose_normal(self, position):
        for die in self.board.dice:
            if die.type == self.board.dice_types.NORMAL:
                return die

    def always_choose_risky(self, position):
        for die in self.board.dice:
            if die.type == self.board.dice_types.RISKY:
                return die

    def random_strategy(self, position):
        return np.random.choice(self.board.dice)

    def optimal_MDP_strategy(self, position):
        type_die = self.board.dice_types(self.optimal_MDP_policy[position])
        for die in self.board.dice:
            if die.type == type_die:
                return die
    
    def optimal_QLearning_strategy(self, position):
        type_die = self.board.dice_types(self.optimal_QLearning_policy[position])
        for die in self.board.dice:
            if die.type == type_die:
                return die

    def risky_then_cautious(self, position):
        if position < self.board.slow_lane.start:
            return self.always_choose_risky()
        elif position < self.board.fast_lane.start:
            return self.always_choose_normal()
        else:
            return self.always_choose_security()