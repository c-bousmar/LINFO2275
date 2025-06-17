from enum import Enum

class DieType(Enum):
    """
    Enumeration representing the types of dice available in the game.
    Each die type corresponds to a different risk/reward profile.
    """
    SECURITY = 1  # Conservative die with safer moves
    NORMAL = 2    # Balanced die with moderate risk
    RISKY = 3     # High-risk die with potential for big moves

class CellType(Enum):
    """
    Enumeration representing the types of cells on the game board.
    Each cell type triggers different behaviors when landed on.
    """
    NORMAL = 0   # Standard cell with no special effect
    RESTART = 1  # Sends the player back to the start
    PENALTY = 2  # Applies a penalty (3 moves backward)
    PRISON = 3   # Does not move next turn
    BONUS = 4    # Grants a bonus (can play again)