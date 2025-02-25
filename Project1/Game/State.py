class State:
    """
    Represents the state of a cell on the game board.
    Combines the cell's position and its type to define its behavior in the game.
    """

    def __init__(self, position, cell_type):
        """
        Initialize a cell state with its position and type.
            - position (int): The index of the cell on the board.
            - cell_type (CellType): The type of the cell (NORMAL, BONUS, PENALTY, PRISON, RESTART).
        """
        self.position = position
        self.cell_type = cell_type