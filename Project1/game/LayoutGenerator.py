from game.Enum import CellType

import numpy as np # type: ignore

class LayoutGenerator:
    
    def __init__(self, event_density):
        bonus_density = event_density[0]
        penalty_density = event_density[1]
        prison_density = event_density[2]
        restart_density = event_density[3]
        self.size = 15
        self.num_bonus = int(self.size * bonus_density)
        self.num_penalty = int(self.size * penalty_density)
        self.num_prison = int(self.size * prison_density)
        self.num_restart = int(self.size * restart_density)
    
    def generate_layout(self):   
        self.layout = np.zeros(self.size, dtype=int)
        special_positions = np.random.choice(range(1, self.size - 1), self.num_bonus + self.num_penalty + self.num_prison + self.num_restart, replace=False)
        self.layout[special_positions[:self.num_bonus]] = CellType.BONUS.value
        self.layout[special_positions[self.num_bonus:self.num_bonus + self.num_penalty]] = CellType.PENALTY.value
        self.layout[special_positions[self.num_bonus + self.num_penalty:self.num_bonus + self.num_penalty + self.num_prison]] = CellType.PRISON.value
        self.layout[special_positions[self.num_bonus + self.num_penalty + self.num_prison:]] = CellType.RESTART.value
        return self.layout