from game.TransitionManager import TransitionManager
from game.BoardGame import BoardGame

import numpy as np # type: ignore
    
def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process using Value Iteration.
    """
    max_iterations = 500
    tol = 1e-9
    
    board = BoardGame(layout, circle)
    tm = TransitionManager(board)

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
                transitions = tm.transition_probabilities(state, die)
                expected_cost = 1.0 + sum(extra + p * V_hat_k[pos] for pos, (p, extra) in transitions.items())
                V_k[die.type.value] = expected_cost

            best_action = min(V_k, key=V_k.get)
            policies[state.position] = best_action
            V_hat_new[state.position] = V_k[best_action]

            delta = max(delta, abs(V_hat_new[state.position] - V_hat_k[state.position]))

        V_hat_k = V_hat_new
        iteration += 1

    return [V_hat_k[:-1], policies[:-1]]