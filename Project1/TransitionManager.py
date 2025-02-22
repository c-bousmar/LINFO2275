from Enum import CellType, PositionType

class TransitionManager:

    def __init__(self, board):
        self.board = board
    
    def transition_probabilities(self, state, die, board):
        transitions = {}
        offsets = [0, 8] if (state.position == PositionType.SLOW_LANE_FIRST_CELL.value - 1) else [0]

        for prob_move, step_size_move in zip(die.probabilities, die.moves):        
            
            for offset in offsets:
                # Get next position
                new_position = self.get_next_position(state.position, step_size_move, offset, board)

                # Case where the event is not triggered
                current_prob, current_extra = transitions.get(new_position, (0, 0))
                transitions[new_position] = (
                    current_prob + (prob_move * (1 - die.probability_triggers)) / len(offsets),
                    current_extra
                )

                # Case where the event is triggered
                new_position_trigger, extra_cost = self.handling_events(board.states[new_position])
                current_prob, current_extra = transitions.get(new_position_trigger, (0, 0))
                transitions[new_position_trigger] = (
                    current_prob + (prob_move * die.probability_triggers) / len(offsets),
                    current_extra + (prob_move * die.probability_triggers * extra_cost) / len(offsets)
                )

        return transitions

    def handling_events(self, state):
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
        # Basic movement
        new_position = position + step_size_move + offset
        
        # Handle special cases
        if offset == PositionType.FAST_LANE_FIRST_CELL.value - PositionType.SLOW_LANE_FIRST_CELL.value + 1:
            new_position -= 1
            if new_position == PositionType.SLOW_LANE_LAST_CELL.value:
                new_position = position
        
        # Handling the gap between the 9_th position and the 14_th position (only one cell)
        if PositionType.SLOW_LANE_FIRST_CELL.value <= position <= PositionType.SLOW_LANE_LAST_CELL.value:
            if not (PositionType.SLOW_LANE_FIRST_CELL.value <= new_position <= PositionType.SLOW_LANE_LAST_CELL.value):
                new_position += PositionType.FAST_LANE_LAST_CELL.value - PositionType.FAST_LANE_FIRST_CELL.value + 1

        # Handling circle (if any) and wrapping around the board
        if new_position > PositionType.FINAL_CELL.value:
            new_position = PositionType.FINAL_CELL.value if not board.circle else new_position - len(board.states)
        
        return new_position