import random as rd

class Dice:
    
    def __init__(self, name, values, probabilities, probability_triggers):
        self.name = name
        self.values = values
        self.probabilities = probabilities
        self.probability_triggers = probability_triggers
        
    def roll(self):
        return rd.choices(self.values, self.probabilities)[0]