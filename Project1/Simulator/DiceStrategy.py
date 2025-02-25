from Algorithms.QLearningDecision import QLearningDecision
from Algorithms.MarkovDecision import markovDecision

from Game.TransitionManager import TransitionManager

import numpy as np

class DiceStrategy:
    """
    Defines various dice selection strategies for a board game. 
    Strategies determine which dice to roll at each position on the board.
    """
    
    def __init__(self, board, strategy_names=None):
        """
        Initializes the DiceStrategy class with available strategies.
        
        Args:
            board (BoardGame): The board game instance.
            strategy_names (list, optional): A list of strategy names to use. Defaults to None (all strategies).
        """
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
        """
        Always selects the security die.
        Returns: Die: The security die.
        """
        for die in self.board.dice:
            if die.type == self.board.dice_types.SECURITY:
                return die

    def always_choose_normal(self, position):
        """
        Always selects the normal die.
        Returns: Die: The normal die.
        """
        for die in self.board.dice:
            if die.type == self.board.dice_types.NORMAL:
                return die

    def always_choose_risky(self, position):
        """
        Always selects the risky die.
        Returns: Die: The risky die.
        """
        for die in self.board.dice:
            if die.type == self.board.dice_types.RISKY:
                return die

    def random_strategy(self, position):
        """
        Selects a die randomly.
        Returns: Die: A randomly chosen die.
        """
        return np.random.choice(self.board.dice)

    def optimal_MDP_strategy(self, position):
        """
        Uses the optimal strategy computed by Markov Decision Process (MDP).
        Returns: Die: The die chosen according to the optimal MDP policy.
        """
        type_die = self.board.dice_types(self.optimal_MDP_policy[position])
        for die in self.board.dice:
            if die.type == type_die:
                return die
    
    def optimal_QLearning_strategy(self, position):
        """
        Uses the optimal strategy computed by Q-Learning.
        Returns: Die: The die chosen according to the optimal Q-learning policy.
        """
        type_die = self.board.dice_types(self.optimal_QLearning_policy[position])
        for die in self.board.dice:
            if die.type == type_die:
                return die

    def risky_then_cautious(self, position):
        """
        Uses a strategy where the player starts by taking risks and later plays cautiously.
        Returns: Die: A die selected based on the position.
        """
        if position < self.board.slow_lane.start:
            return self.always_choose_risky()
        elif position < self.board.fast_lane.start:
            return self.always_choose_normal()
        else:
            return self.always_choose_security()