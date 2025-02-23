from Game.TransitionManager import TransitionManager
from Game.BoardGame import BoardGame

import numpy as np # type: ignore

# def QLearningDecision(layout, circle, episodes=1000, alpha=0.1, gamma=0.9, display_board=False):
#     """
#     Implements Q-learning to estimate the optimal policy.

#     :param layout: List defining the board layout.
#     :param circle: Boolean indicating whether the board wraps around.
#     :param alpha: Learning rate.
#     :param gamma: Discount factor for future rewards.
#     :param episodes: Number of learning episodes.
#     :return: Q_hat matrix with estimated Q-values.
#     """
#     board = BoardGame(layout, circle)
#     if display_board: board.display_board()
#     tm = TransitionManager(board)

#     Q_hat = np.zeros((len(board.states), len(board.dice)), dtype=float)

#     # Q_hat(k,a) = Q_hat(k,a) + alpha(t) [c(a|k) + V_star(k`) - Q_hat(k,a)],
#     #   where V_star(k`) = min(Q_hat(k`, a`))
#     for _ in range(episodes):
#         state = 0

#         while state < len(board.states) - 1:
#             action = np.random.choice(len(board.dice))

#             transitions = tm.transition_probabilities(board.states[state], board.dice[action])
#             expected_value = sum(prob * (reward + gamma * np.min(Q_hat[next_state, :]))
#                                  for next_state, (prob, reward) in transitions.items())
            
#             Q_hat[state, action] += alpha * (expected_value - Q_hat[state, action])
#             state = np.random.choice(list(transitions.keys()), p=[prob for prob, _ in transitions.values()])
    
#     expectations = np.min(Q_hat, axis=1)[:-1]
#     die_optimal = np.argmin(Q_hat, axis=1)[:-1] + 1  # Convert to 1-based indexing (1 for "security", 2 for "normal", 3 for "risky")
    
#     return [expectations, die_optimal]

def QLearningDecision(layout, circle, epochs=1000, alpha=0.1, gamma=0.9, epsilon=0.1, display_board=False):
    """
    Implements Q-learning to estimate the optimal policy with ε-greedy exploration.
    
    :param layout: List defining the board layout.
    :param circle: Boolean indicating whether the board wraps around.
    :param alpha: Learning rate.
    :param gamma: Discount factor for future rewards.
    :param epsilon: Exploration rate (for ε-greedy strategy).
    :param epochs: Number of learning episodes.
    :param display_board: Boolean to display the board during learning.
    :return: Q_hat matrix with estimated Q-values.
    """
    board = BoardGame(layout, circle)
    tm = TransitionManager(board)
    
    if display_board: 
        board.display_board()

    Q_hat = np.zeros((len(board.states), len(board.dice)), dtype=float)

    for _ in range(epochs):
        position = board.start_cell

        while position < board.last_cell:
            # Exploration vs Exploitation: ε-greedy action selection
            if np.random.rand() < epsilon:  # Exploration
                die = np.random.choice(board.dice)
            else:  # Exploitation: choose the die with the highest Q-value
                die_id = np.argmax(Q_hat[position, :])
                die = board.dice[die_id]
            
            die_id = die.type.value - 1 # Adjust for indexing (0-based)
            transitions = tm.transition_probabilities(board.states[position], die)
            expected_value = sum(prob * (reward + gamma * np.max(Q_hat[next_position, :]))
                                 for next_position, (prob, reward) in transitions.items())
            Q_hat[position, die_id] += alpha * (expected_value - Q_hat[position, die_id])
            position = np.random.choice(list(transitions), p=[prob for prob, _ in transitions.values()])

    expectations = np.min(Q_hat, axis=1)[:-1]
    die_optimal = np.argmin(Q_hat, axis=1)[:-1] + 1 # Convert to 1-based indexing
    
    return [expectations, die_optimal]