from enum import Enum

class DieType(Enum):
	SECURITY = 1
	NORMAL = 2
	RISKY = 3

class PositionType(Enum):
    START_CELL = 1
    FINAL_CELL = 14
    SLOW_LANE_FIRST_CELL = 3
    SLOW_LANE_LAST_CELL = 9
    FAST_LANE_FIRST_CELL = 10
    FAST_LANE_LAST_CELL = 13

class CellType(Enum):
    NORMAL = 0
    RESTART = 1
    PENALTY = 2
    PRISON = 3
    BONUS = 4
    
class CellTypeEmoji(Enum):
    NORMAL = "X"
    RESTART = "🔄"
    PENALTY = "⬅"
    PRISON = "⏳"
    BONUS = "⭐"