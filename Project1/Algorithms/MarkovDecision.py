from Game.TransitionManager import TransitionManager
from Game.BoardGame import BoardGame

import numpy as np

def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process (MDP) using Value Iteration.
    This algorithm computes the optimal policy and value function for each state on the board.

    Args:
        layout (list): The layout of the board, specifying the type of each cell.
        circle (bool): Indicates whether the board is circular (unused in this implementation).

    Returns:
        list: A list containing two elements:
              - V_hat_k: The value function for each state (excluding the final state).
              - policies: The optimal policy (action to take) for each state (excluding the final state).
    """
    # Parameters for value iteration
    max_iterations = 500
    tol = 1e-9

    # Initialize the board and transition manager
    board = BoardGame(layout, circle)
    tm = TransitionManager(board)

    # Initialize value function and policy arrays
    V_hat_k = np.zeros(len(board.states), dtype=float)
    policies = np.zeros(len(board.states), dtype=int)

    iteration = 0
    delta = float('inf')

    # Value iteration loop
    while delta > tol and iteration < max_iterations:
        delta = 0
        V_hat_new = np.copy(V_hat_k)

        # Iterate over all states except the final state
        for state in board.states[:-1]:
            V_k = {}  # Dictionary to store expected costs for each action (die type)

            # Evaluate each possible action (die type)
            for die in board.dice:
                # Get transition probabilities and costs for the current state and die
                transitions = tm.transition_probabilities(state, die)
                # Compute the expected cost for this action
                expected_cost = 1.0 + sum(
                    extra + p * V_hat_k[pos] for pos, (p, extra) in transitions.items()
                )
                V_k[die.type.value] = expected_cost

            # Choose the action with the minimum expected cost
            best_action = min(V_k, key=V_k.get)
            policies[state.position] = best_action
            V_hat_new[state.position] = V_k[best_action]

            # Track the maximum change in value function
            delta = max(delta, abs(V_hat_new[state.position] - V_hat_k[state.position]))

        V_hat_k = V_hat_new
        iteration += 1

    return [V_hat_k[:-1], policies[:-1]]