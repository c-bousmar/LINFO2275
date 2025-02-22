from Enum import CellType, PositionType

class TransitionManager:
    """
    Manages state transitions and movement rules on a board game.

    Attributes:
        board (Board): The game board containing layout and rules.

    Methods:
        transition_probabilities(state, die, board):
            Computes the transition probabilities for a given state based on dice rolls and board rules.

        handling_events(state):
            Handles events based on the type of cell the player is on, updating position and status.

        get_next_position(position, step_size_move, offset, board):
            Calculates the next position of a player on the board based on movement rules, considering board constraints.
    """
    def __init__(self, board):
        self.board = board
    
    
    def transition_probabilities(self, state, die, board):
        """
        Computes the transition probabilities for a given state based on the dice roll and board rules.

        This function calculates the probability distribution of possible next positions 
        based on the outcome of rolling a die, considering special events and board structure.

        Parameters:
            state (State): The current game state, including the player's position.
            die (Die): The die used for movement, containing:
                - `probabilities`: Probabilities of each possible movement step.
                - `moves`: Corresponding movement steps.
                - `probability_triggers`: Probability of triggering a special event.
            board (Board): The game board, which defines cell types and special behaviors.

        Returns:
            dict[int, tuple[float, float]]: A dictionary mapping possible next positions to:
                - The probability of reaching that position.
                - The expected additional cost from triggered events.

        Movement Mechanics:
            - The function iterates through all possible dice outcomes and calculates
            the resulting position.
            - If the player is at the cell before the slow lane's first cell, an additional 
            offset (8 cells forward) is considered.
            - The function accounts for both regular movement and movement when an event 
            is triggered.

        Event Handling:
            - If an event is triggered upon landing on a cell, the probability is split
            accordingly between regular movement and movement affected by events.
            - The expected additional cost from events is accumulated in the probability 
            distribution.

        Board Constraints:
            - Movement respects board boundaries and special lane transitions.
            - The board structure and special event mechanics influence the probability 
            distribution of possible transitions.
        """
        transitions = {}
        offsets = [0, 8] if (state.position == PositionType.SLOW_LANE_FIRST_CELL.value - 1) else [0]

        for prob_move, step_size_move in zip(die.probabilities, die.moves):        
            
            for offset in offsets:
                new_position = self.get_next_position(state.position, step_size_move, offset, board)

                current_prob, current_extra = transitions.get(new_position, (0, 0))
                transitions[new_position] = (
                    current_prob + (prob_move * (1 - die.probability_triggers)) / len(offsets),
                    current_extra
                )

                new_position_trigger, extra_cost = self.handling_events(board.states[new_position])
                current_prob, current_extra = transitions.get(new_position_trigger, (0, 0))
                transitions[new_position_trigger] = (
                    current_prob + (prob_move * die.probability_triggers) / len(offsets),
                    current_extra + (prob_move * die.probability_triggers * extra_cost) / len(offsets)
                )
        return transitions


    def handling_events(self, state):
        """
        Handles events based on the type of cell the player is on.

        This function determines the player's new position and any status effects 
        based on the `cell_type` of the current state.

        Parameters:
            state (State): The current game state, which includes the player's position 
                        and the type of cell they are on.

        Returns:
            tuple[int, int]: A tuple containing:
                - The updated position of the player.
                - A status indicator:
                    - `0` for no special effect.
                    - `1` if the player is in prison.
                    - `-1` if the player receives a bonus.

        Cell Type Effects:
            - `CellType.RESTART`: Resets the position to `0` and clears any status.
            - `CellType.PENALTY`: Moves the player back 3 positions (minimum `0`).
            - `CellType.PRISON`: Keeps the player at their current position with a status of `1` (indicating prison).
            - `CellType.BONUS`: Keeps the player at their current position but applies a status of `-1` (indicating a bonus).
            - Default case: No change in position or status.
        """
        match state.cell_type:
            case CellType.RESTART:
                return 0, 0
            case CellType.PENALTY:
                return max(0, state.position - 3), 0
            case CellType.PRISON:
                return state.position, 1
            case CellType.BONUS:
                return state.position, -1
        return state.position, 0


    def get_next_position(self, position, step_size_move, offset, board):
        """
        Calculates the next position of a player on the board based on movement rules.

        This function updates the player's position by considering their step size, 
        additional offset, and specific board constraints such as lane transitions 
        and circular board behavior.

        Parameters:
            position (int): The current position of the player on the board.
            step_size_move (int): The number of steps the player moves forward.
            offset (int): Additional movement offset, which may result from special 
                        effects or lane transitions.
            board (Board): The game board, which contains information about its structure, 
                        including whether it wraps around (circular board).

        Returns:
            int: The updated position of the player after applying movement rules.

        Special Cases:
            - **Fast Lane to Slow Lane Transition**:
            - If the offset corresponds to the transition from the fast lane to the slow lane, 
                the movement is adjusted accordingly.
            - If the new position lands exactly on `SLOW_LANE_LAST_CELL`, 
                the player remains at the original position.

            - **Handling Gaps between Lanes**:
            - If the player is in the slow lane but moves out of its range, they are placed 
                in the corresponding position in the fast lane.

            - **Wrapping Around (Circular Board)**:
            - If the new position exceeds `FINAL_CELL`, the position wraps around 
                based on the total number of board states if the board is circular.
        """
        new_position = position + step_size_move + offset
        
        if offset == PositionType.FAST_LANE_FIRST_CELL.value - PositionType.SLOW_LANE_FIRST_CELL.value + 1:
            new_position -= 1
            if new_position == PositionType.SLOW_LANE_LAST_CELL.value:
                new_position = position
        
        if PositionType.SLOW_LANE_FIRST_CELL.value <= position <= PositionType.SLOW_LANE_LAST_CELL.value:
            if not (PositionType.SLOW_LANE_FIRST_CELL.value <= new_position <= PositionType.SLOW_LANE_LAST_CELL.value):
                new_position += PositionType.FAST_LANE_LAST_CELL.value - PositionType.FAST_LANE_FIRST_CELL.value + 1

        if new_position > PositionType.FINAL_CELL.value:
            new_position = PositionType.FINAL_CELL.value if not board.circle else new_position - len(board.states)
        return new_position