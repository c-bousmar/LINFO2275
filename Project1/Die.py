import numpy as np

class Die:
    
    def __init__(self, type_die, moves, probabilities, probability_triggers):
        self.type = type_die
        self.moves = moves
        self.probabilities = probabilities
        self.probability_triggers = probability_triggers
        
    def roll(self):
        return np.random.choice(self.moves, p=self.probabilities)
    
    def is_triggering_event(self):
        return np.random.choice([True, False], p=[self.probability_triggers, 1 - self.probability_triggers])