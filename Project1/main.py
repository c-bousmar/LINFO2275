from BoardGame import BoardGame

import numpy as np

DICE_OPTIONS = {
    1: [0, 1],  # Security Dice
    2: [0, 1, 2],  # Normal Dice
    3: [0, 1, 2, 3]  # Risky Dice
}

DICE_PROBABILITIES = { # Allows to get different probabilities for value on each dice
    1: [0.5, 0.5],
    2: [1/3, 1/3, 1/3],
    3: [1/4, 1/4, 1/4, 1/4]
}

def transition_probabilities(k, a, layout, is_circle):
    """
    Computes the transition probabilities for moving from state `k` to state `k'` given action `a`.

    Parameters:
    -----------
    k : int
        The current state on the board (0-based index).
    
    a : int
        The action represented by the dice choice (1 for security, 2 for normal, 3 for risky).
    
    layout : numpy.ndarray
        A 1D array of length 15 representing the game board. Values indicate:
        - 0: Normal square
        - 1: Restart trap
        - 2: Penalty trap
        - 3: Prison trap
        - 4: Bonus square
    
    is_circle : bool
        If `True`, the game requires landing **exactly** on the goal square (index 14).
        If the move exceeds square 14, the player continues from the beginning.

    Returns:
    --------
    dict
        A dictionary mapping each possible next state `k'` to its probability of occurrence:
        `{k': p(k' | k, a)}`.
    """
    transistions  = {}
    
    for dice_index, dice_value in enumerate(DICE_OPTIONS[a]):
        p_kʹ = k + dice_value

        if is_circle and p_kʹ > 14:
            p_kʹ = 0 + (p_kʹ - 14)
        elif p_kʹ > 14:
            p_kʹ = 14

        if layout[p_kʹ] == 1:
            p_kʹ = 0

        elif layout[p_kʹ] == 2:
            p_kʹ = max(0, p_kʹ - 3)

        elif layout[p_kʹ] == 3:
            p_kʹ = k
        
        p_kʹ_given_k_and_a = DICE_PROBABILITIES[a][dice_index]
        
        transistions[p_kʹ] = transistions.get(p_kʹ, 0) + p_kʹ_given_k_and_a

    return transistions


def markovDecision(layout, circle):
    """
    Solves the Markov Decision Process using Value Iteration to determine the optimal dice choice for each state.

    Parameters:
    -----------
    layout : numpy.ndarray
        The game board layout with traps and bonuses.
    
    circle : bool
        If True, the player must land exactly on the goal square.

    Returns:
    --------
    list
        A list [Expec, Dice] where:
        - Expec (numpy.ndarray) contains the expected cost (number of turns) for each state.
        - Dice (numpy.ndarray) contains the optimal dice choice for each state.
    """    
    game = BoardGame(layout, circle)
    size = game.size

    max_iterations = 1000
    tol = 1e-9
    iteration = 0
    delta = float("inf")

    V_hat_k = np.zeros(size, dtype=float)
    policies = np.zeros(size, dtype=int)

    while delta > tol:
        delta = 0
        V_hat_new = np.copy(V_hat_k)

        for k in range(size - 1):
            
            V_k = {}
            c_k = abs(14 - k) # c(a|k) := distance to the last square

            for a in [1, 2, 3]:
                transitions = transition_probabilities(k, a, layout, circle)
                # V̂(k) ← min_{a} { c(a|k) + ∑ p(k'|k, a) V̂(k') }
                V_k[a] = c_k + sum(p_kʹ_given_k_and_a * V_hat_k[kʹ] for kʹ, p_kʹ_given_k_and_a in transitions.items())

            policies[k] = min(V_k, key=V_k.get)
            V_hat_k[k] = V_k[policies[k]]

            delta = max(delta, abs(V_hat_k[k] - V_hat_new[k]))

        V_hat_k = V_hat_new

        if iteration >= max_iterations:
            break
        iteration += 1

    return [V_hat_k[:-1], policies[:-1]]



if __name__ == '__main__':
    
    layout = np.array([0] * 15)
    circle = False

    decisions = markovDecision(layout, circle)
    
    print("Expectation: ", decisions[0])
    print("Dice: ", decisions[1])