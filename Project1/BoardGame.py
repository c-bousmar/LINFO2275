from State import State
from Die import Die
from Enum import DieType, PositionType, CellType, CellTypeEmoji

from itertools import chain

class BoardGame:
    
    def __init__(self, layout, circle):
        self.layout = layout
        self.circle = circle
        self.size = len(layout)
        position_map = {position.value: position.name for position in PositionType}
        self.states = [ State(i, position_map.get(i, "INTERMEDIATE_CELL"), CellType(cell)) for i, cell in enumerate(layout) ]
        self.final_state = self.states[self.size - 1]
        self.dice   = [ Die(type_die=DieType.SECURITY, moves=[0, 1], probabilities=[1/2, 1/2], probability_triggers=0.0),
                        Die(type_die=DieType.NORMAL, moves=[0, 1, 2], probabilities=[1/3, 1/3, 1/3], probability_triggers=1/2),
                        Die(type_die=DieType.RISKY, moves=[0, 1, 2, 3], probabilities=[1/4, 1/4, 1/4, 1/4], probability_triggers=1.0) ]
    
    def display_board(self):
        layout = "Board :\n"
        layout += " _________________________________________________________________\n"
        layout += "|"
        for i in chain(
            range(0, PositionType.FAST_LANE_FIRST_CELL.value),
            range(PositionType.FAST_LANE_LAST_CELL.value + 1, PositionType.SLOW_LANE_LAST_CELL.value + 1),
            range(PositionType.FINAL_CELL.value, PositionType.FINAL_CELL.value + 1)
        ):
            print(i)
            emoji = CellTypeEmoji[self.states[i].cell_type.name].value
            nb_whitespace = 2 if (emoji == "X" or emoji == "⬅") else 1
            layout += f"  {emoji}" + " " * nb_whitespace + "|"
        layout += "\n|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|\n"
        layout += "               \\                                              / \n"
        layout += "           _____\\____________________________________________/____\n"
        layout += "          |"
        for i in range(PositionType.FAST_LANE_FIRST_CELL.value, PositionType.FAST_LANE_LAST_CELL.value + 1):
            emoji = CellTypeEmoji[self.states[i].cell_type.name].value
            nb_whitespace = 6 if (emoji == "X" or emoji == "⬅") else 5
            layout += f"      {emoji}" + " " * nb_whitespace + "|"
        layout += "\n          |_____________|_____________|_____________|_____________|\n"
        print(layout)