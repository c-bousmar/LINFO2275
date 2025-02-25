from .Enum import DieType, CellType
from .State import State
from .Die import Die

from itertools import chain

class BoardGame:
    """
    Represents a board game with customizable layout and game mechanics.
    Handles board state management, dice configuration, and visual display.
    """
    
    def __init__(self, layout, circle):
        """
        Initialize board game with:
            - layout: List of 15 values representing board cells (0 = Normal,
                      1 = Restart, Trap; 2 = Penalty Trap, 3 = Prison Trap, 4 = Bonus)
            - circle: Boolean indicating circular board behavior
        """
        self.layout = layout
        self.circle = circle
        
        # Validate board size (must have exactly 15 cells)
        assert len(layout) == 15
        
        # Cell configuration
        self.cell_types = CellType
        self.cell_emojis = { # Mapping of cell types to display symbols
            "NORMAL" : "X",
            "RESTART" : "🔄",
            "PENALTY" : "⬅",
            "PRISON" : "⏳",
            "BONUS" : "⭐"
        }

        # Board position configuration
        self.start_cell = 0
        self.slow_lane = range(3, 10)
        self.fast_lane = range(10, 14)
        self.last_cell = 14

        # Initialize game states for each cell
        self.states = [ State(i, self.cell_types(cell)) for i, cell in enumerate(layout) ]
        self.final_state = self.states[self.last_cell]
        
        # Configure game dice with different behaviors
        self.dice_types = DieType
        self.dice   = [ Die(type_die=self.dice_types.SECURITY, moves=[0, 1], probabilities=[1/2, 1/2], probability_triggers=0.0),
                        Die(type_die=self.dice_types.NORMAL, moves=[0, 1, 2], probabilities=[1/3, 1/3, 1/3], probability_triggers=1/2),
                        Die(type_die=self.dice_types.RISKY, moves=[0, 1, 2, 3], probabilities=[1/4, 1/4, 1/4, 1/4], probability_triggers=1.0) ]

    def display_board(self):
        """Generate and print ASCII-art representation of the game board"""
        layout = "Board :\n"
        layout += " _________________________________________________________________\n"
        layout += "|"
        # Build top section (non-fast lane cells)
        for i in chain(
            range(0, self.fast_lane.start),
            range(self.fast_lane.stop, self.slow_lane.stop),
            range(self.last_cell, self.last_cell + 1)
        ):
            emoji = self.cell_emojis[self.states[i].cell_type.name]
            nb_whitespace = 2 if (emoji == "X" or emoji == "⬅") else 1
            layout += f"  {emoji}" + " " * nb_whitespace + "|"
        # Build middle section separators
        layout += "\n|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|_____|\n"
        layout += "               \\                                              / \n"
        layout += "           _____\\____________________________________________/____\n"
        layout += "          |"
        # Build fast lane section
        for i in self.fast_lane:
            emoji = self.cell_emojis[self.states[i].cell_type.name]
            nb_whitespace = 6 if (emoji == "X" or emoji == "⬅") else 5
            layout += f"      {emoji}" + " " * nb_whitespace + "|"
        layout += "\n          |_____________|_____________|_____________|_____________|\n"
        print(layout)