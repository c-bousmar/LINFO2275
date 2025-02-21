from BoardGame import BoardGame
from Enum import PositionType, CellType

import numpy as np

def transition_probabilities(state, die, board):
    
    transitions = {}

    for idx_move, step_size_move in enumerate(die.moves):
        
        offsets = [0]
        
        # Handling the case where we have the fast lane and the slow lane
        if state.position_type == PositionType.SLOW_LANE_FIRST_CELL.value - 1:
            offsets.append(8)
        
        for offset in offsets:
            new_position = state.position + step_size_move + offset

            # Handling circle (if any) and wrapping around the board
            if new_position > PositionType.FINAL_CELL.value:
                new_position = PositionType.FINAL_CELL.value if not board.circle else new_position - len(board.states)
        
            new_state = board.states[new_position]
            
            # Handling traps and special squares
            extra_cost = 0
            if new_state.cell_type == CellType.RESTART and die.is_triggering_event():
                new_position = 0
            elif new_state.cell_type == CellType.PENALTY and die.is_triggering_event():
                new_position = max(0, new_position - 3)
            elif new_state.cell_type == CellType.PRISON and die.is_triggering_event():
                extra_cost = 1
            elif new_state.cell_type == CellType.BONUS and die.is_triggering_event():
                extra_cost = -1

            prob, cost = transitions.get(new_position, (0, 0))
            transitions[new_position] = (prob + die.probabilities[idx_move] / len(offsets), cost + extra_cost)

    return transitions

def get_cost_to_goal_state(position):
    if PositionType.FAST_LANE_FIRST_CELL.value <= position <= PositionType.FAST_LANE_LAST_CELL.value:
        return PositionType.FINAL_CELL.value - position
    else:
        return PositionType.SLOW_LANE_LAST_CELL.value - position + 1
    
def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process using Value Iteration.
    """
    max_iterations = 50000
    tol = 1e-9
    
    board = BoardGame(layout, circle)

    V_hat_k = np.zeros(len(board.states), dtype=float)
    policies = np.zeros(len(board.states), dtype=int)
    
    iteration = 0
    delta = float('inf')
    
    while delta > tol and iteration < max_iterations:
        delta = 0
        V_hat_new = np.copy(V_hat_k)

        for state in board.states[:-1]:
            V_k = {}
            cost = get_cost_to_goal_state(state.position)

            for die in board.dice:
                transitions = transition_probabilities(state, die, board)
                expected_cost = cost + sum(p * (V_hat_k[pos] + extra) for pos, (p, extra) in transitions.items())
                V_k[die.type.value] = expected_cost

            best_action = min(V_k, key=V_k.get)
            policies[state.position] = best_action
            V_hat_new[state.position] = V_k[best_action]

            delta = max(delta, abs(V_hat_new[state.position] - V_hat_k[state.position]))

        V_hat_k = V_hat_new
        iteration += 1

    return [V_hat_k[:-1], policies[:-1]]