from .Enum import CellType
import numpy as np

class LayoutGenerator:
    """
    Generates a randomized game board layout based on specified event densities.
    Ensures a balanced distribution of special cells (bonus, penalty, prison, restart).
    """

    def __init__(self, event_density):
        """
        Initialize the layout generator with event densities.
            - event_density (list): A list of densities for [bonus, penalty, prison, restart] events.
                                    Each density is a float between 0 and 1, representing the proportion
                                    of the board to allocate to that event type.
        """
        # Extract densities for each event type
        bonus_density = event_density[0]
        penalty_density = event_density[1]
        prison_density = event_density[2]
        restart_density = event_density[3]

        # Board size is fixed at 15 cells
        self.size = 15

        # Calculate the number of cells for each event type based on density
        self.num_bonus = int(self.size * bonus_density)
        self.num_penalty = int(self.size * penalty_density)
        self.num_prison = int(self.size * prison_density)
        self.num_restart = int(self.size * restart_density)

    def generate_layout(self):
        """
        Generates a randomized board layout with special cells distributed across the board.
        Returns: np.ndarray: An array representing the board layout, where each cell is assigned a CellType value.
        """
        # Initialize layout with NORMAL cells (CellType.NORMAL.value = 0)
        self.layout = np.zeros(self.size, dtype=int)

        # Randomly select positions for special cells, avoiding the last cell
        special_positions = np.random.choice(
            range(self.size - 1),  # Avoid end cell
            self.num_bonus + self.num_penalty + self.num_prison + self.num_restart,
            replace=False  # Ensure no duplicate positions
        )

        # Assign event types to the selected positions
        self.layout[special_positions[:self.num_bonus]] = CellType.BONUS.value  # Bonus cells
        self.layout[special_positions[self.num_bonus:self.num_bonus + self.num_penalty]] = CellType.PENALTY.value  # Penalty cells
        self.layout[special_positions[self.num_bonus + self.num_penalty:self.num_bonus + self.num_penalty + self.num_prison]] = CellType.PRISON.value  # Prison cells
        self.layout[special_positions[self.num_bonus + self.num_penalty + self.num_prison:]] = CellType.RESTART.value  # Restart cells

        return self.layout