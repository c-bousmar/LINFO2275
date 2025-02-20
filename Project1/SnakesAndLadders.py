from State import State
from Dice import Dice

STATE_TYPE = {0 : "basic", 1 : "restart", 2 : "penalty", 3 : "prison", 4 : "bonus"}
SYMBOLS = {0: "X", 1: "🔄", 2: "⬅", 3: "⏳", 4: "⭐"}

class SnakesAndLaddersMDP:
    
    def __init__(self, layout, circle):
        self.layout = layout
        self.circle = circle
        
        self.states = []
        for i, cell in enumerate(self.layout):
            state_type = STATE_TYPE.get(cell, None)
            is_junction = i == 2
            self.states.append(State(i, state_type, is_junction))
    
        self.dices = [Dice("security", [0, 1], [1/2, 1/2], 0.0),
                      Dice("normal", [0, 1, 2], [1/3, 1/3], 1/2),
                      Dice("risky", [0, 1, 2, 3], [1/4, 1/4, 1/4, 1/4], 1.0)]
    
    def display_board(self):
        layout = "Board :\n"
        layout += " _________________________________________________________________\n"
        layout += "|"
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14]:
            nb_whitespace = 2 if (SYMBOLS[self.layout[i]] == "X" or SYMBOLS[self.layout[i]] == "⬅") else 1
            layout += f"  {SYMBOLS[self.layout[i]]}" + " " * nb_whitespace + "|"
        layout += "\n|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|\n"
        layout += "               \\                                              / \n"
        layout += "           _____\\____________________________________________/____\n"
        layout += "          |"
        for i in [10, 11, 12, 13]:
            nb_whitespace = 6 if (SYMBOLS[self.layout[i]] == "X" or SYMBOLS[self.layout[i]] == "⬅") else 5
            layout += f"      {SYMBOLS[self.layout[i]]}" + " " * nb_whitespace + "|"
        layout += "\n          |_____________|_____________|_____________|_____________|\n"
        print(layout)
        
        
game = SnakesAndLaddersMDP([4, 4, 1, 3, 1, 2, 1, 2, 0, 4, 0, 3, 0, 2, 2], False)
game.display_board()