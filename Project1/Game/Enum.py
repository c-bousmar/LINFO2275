from enum import Enum

class DieType(Enum):
	SECURITY = 1
	NORMAL = 2
	RISKY = 3

class CellType(Enum):
    NORMAL = 0
    RESTART = 1
    PENALTY = 2
    PRISON = 3
    BONUS = 4