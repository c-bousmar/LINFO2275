from Enum import PositionType, CellType
from BoardGame import BoardGame

import numpy as np

def transition_probabilities(state, die, board):
    transitions = {}
    offsets = [0, 8] if (state.position == PositionType.SLOW_LANE_FIRST_CELL.value - 1) else [0]

    for prob_move, step_size_move in zip(die.probabilities, die.moves):        
        
        for offset in offsets:
            # Get next position
            new_position = get_next_position(state.position, step_size_move, offset, board)

            # Case where the event is not triggered
            current_prob, current_extra = transitions.get(new_position, (0, 0))
            transitions[new_position] = (
                current_prob + (prob_move * (1 - die.probability_triggers)) / len(offsets),
                current_extra
            )

            # Case where the event is triggered
            new_position_trigger, extra_cost = handling_events(board.states[new_position])
            current_prob, current_extra = transitions.get(new_position_trigger, (0, 0))
            transitions[new_position_trigger] = (
                current_prob + (prob_move * die.probability_triggers) / len(offsets),
                current_extra + (prob_move * die.probability_triggers * extra_cost) / len(offsets)
            )

    return transitions

def handling_events(state):
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

def get_next_position(position, step_size_move, offset, board):
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
    
def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process using Value Iteration.
    """
    max_iterations = 1000
    tol = 1e-9
    
    board = BoardGame(layout, circle)
    board.display_board()

    V_hat_k = np.zeros(len(board.states), dtype=float)
    policies = np.zeros(len(board.states), dtype=int)
    
    iteration = 0
    delta = float('inf')
    
    while delta > tol and iteration < max_iterations:
        delta = 0
        V_hat_new = np.copy(V_hat_k)

        for state in board.states[:-1]:
            V_k = {}
            
            for die in board.dice:
                transitions = transition_probabilities(state, die, board)
                expected_cost = 1.0 + sum(extra + p * V_hat_k[pos] for pos, (p, extra) in transitions.items())
                V_k[die.type.value] = expected_cost

            best_action = min(V_k, key=V_k.get)
            policies[state.position] = best_action
            V_hat_new[state.position] = V_k[best_action]

            delta = max(delta, abs(V_hat_new[state.position] - V_hat_k[state.position]))

        V_hat_k = V_hat_new
        iteration += 1

    return [V_hat_k[:-1], policies[:-1]]


if __name__ == '__main__':
    
    # layout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # result = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  
    # layout = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    # result = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
    
    # layout = [0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0]
    # result = [2, 1, 2, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    
    # layout = [0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 4, 4, 4, 4, 0]
    # result = [2, 1, 3, 3, 2, 3, 2, 1, 1, 1, 3, 3, 3, 3]
    
    # layout = [0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0]
    # result = [3, 3, 2, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1]
    
    # layout = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # result = [2, 1, 3, 3, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3]
    
    # layout = [0, 1, 3, 4, 2, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0]
    # result = [3, 2, 1, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1]
    
    # layout = [0, 2, 3, 2, 0, 2, 2, 0, 1, 0, 0, 3, 1, 3, 0]
    # result = [2, 1, 2, 2, 3, 1, 1, 1, 1, 3, 1, 1, 1, 3]
    
    # layout = [0, 0, 3, 1, 1, 3, 2, 2, 4, 0, 4, 4, 0, 0, 0]
    # result = [2 1 1 1 1 1 1 1 3 3 1 1 3 3]
    
    layout = [0, 0, 3, 1, 1, 3, 2, 2, 4, 0, 4, 4, 0, 0, 0]
    circle = True
    expectations, die_optimal = markovDecision(layout, circle)
    print(expectations)
    # print(die_optimal)
    print(die_optimal[:10])
    print("     " + f'{die_optimal[10:15]}')