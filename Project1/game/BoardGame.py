from game.Enum import DieType, CellType
from game.State import State
from game.Die import Die

from itertools import chain

class BoardGame:
    
    def __init__(self, layout, circle):
        self.layout = layout
        assert len(layout) == 15 # The length of the board must be equal to 15
        self.circle = circle
        
        self.dice_types = DieType
        self.cell_types = CellType
        self.cell_emojis = {"NORMAL" : "X", "RESTART" : "🔄", "PENALTY" : "⬅", "PRISON" : "⏳", "BONUS" : "⭐"}

        self.start_cell = 0
        self.slow_lane = range(3, 10)
        self.fast_lane = range(10, 14)
        self.last_cell = 14

        self.states = [ State(i, self.cell_types(cell)) for i, cell in enumerate(layout) ]
        self.final_state = self.states[self.last_cell]
        
        self.dice   = [ Die(type_die=self.dice_types.SECURITY, moves=[0, 1], probabilities=[1/2, 1/2], probability_triggers=0.0),
                        Die(type_die=self.dice_types.NORMAL, moves=[0, 1, 2], probabilities=[1/3, 1/3, 1/3], probability_triggers=1/2),
                        Die(type_die=self.dice_types.RISKY, moves=[0, 1, 2, 3], probabilities=[1/4, 1/4, 1/4, 1/4], probability_triggers=1.0) ]

    def display_board(self):
        layout = "Board :\n"
        layout += " _________________________________________________________________\n"
        layout += "|"
        for i in chain(
            range(0, self.fast_lane.start),
            range(self.fast_lane.stop, self.slow_lane.stop),
            range(self.last_cell, self.last_cell + 1)
        ):
            emoji = self.cell_emojis[self.states[i].cell_type.name]
            nb_whitespace = 2 if (emoji == "X" or emoji == "⬅") else 1
            layout += f"  {emoji}" + " " * nb_whitespace + "|"
        layout += "\n|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|\n"
        layout += "               \\                                              / \n"
        layout += "           _____\\____________________________________________/____\n"
        layout += "          |"
        for i in self.fast_lane:
            emoji = self.cell_emojis[self.states[i].cell_type.name]
            nb_whitespace = 6 if (emoji == "X" or emoji == "⬅") else 5
            layout += f"      {emoji}" + " " * nb_whitespace + "|"
        layout += "\n          |_____________|_____________|_____________|_____________|\n"
        print(layout)