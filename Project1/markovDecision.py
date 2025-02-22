from BoardGame import BoardGame
from TransitionManager import TransitionManager

import numpy as np
    
def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process using Value Iteration.
    """
    max_iterations = 1000
    tol = 1e-9
    
    board = BoardGame(layout, circle)
    tm = TransitionManager(board)
    # board.display_board()

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
                transitions = tm.transition_probabilities(state, die, board)
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