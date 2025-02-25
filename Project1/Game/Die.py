import numpy as np

class Die:
    """
    Represents a game die with some behavior.
    Handles move outcomes and event triggering probabilities.
    """
    
    def __init__(self, type_die, moves, probabilities, probability_triggers):
        """
        Initialize a die with:
            - type_die: Type of die (e.g., SECURITY, NORMAL, RISKY)
            - moves: List of possible move outcomes
            - probabilities: List of probabilities corresponding to each move
            - probability_triggers: Probability of triggering a special event
        """
        self.type = type_die
        self.moves = moves
        self.probabilities = probabilities
        self.probability_triggers = probability_triggers

    def roll(self):
        """
        Simulate a die roll.
        Returns a random move based on the die's probability distribution.
        """
        return np.random.choice(self.moves, p=self.probabilities)

    def is_triggering_event(self):
        """
        Check if a special event is triggered.
        Returns True if the event is triggered, False otherwise.
        """
        return np.random.choice(
            [True, False], 
            p=[self.probability_triggers, 1 - self.probability_triggers]
        )