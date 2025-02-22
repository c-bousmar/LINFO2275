from BoardGame import BoardGame
import numpy as np
from TransitionManager import TransitionManager


def QLearningDecision(layout, circle, episodes=1000, alpha=0.1, gamma=0.9, is_test=False):
    """
    Implements Q-learning to estimate the optimal policy.

    :param layout: List defining the board layout.
    :param circle: Boolean indicating whether the board wraps around.
    :param alpha: Learning rate.
    :param gamma: Discount factor for future rewards.
    :param episodes: Number of learning episodes.
    :return: Q_hat matrix with estimated Q-values.
    """
    board = BoardGame(layout, circle)
    if is_test: board.display_board()
    tm = TransitionManager(board)

    Q_hat = np.zeros((len(board.states), len(board.dice)), dtype=float)

    # Q_hat(k,a) = Q_hat(k,a) + alpha(t) [c(a|k) + V_star(k`) - Q_hat(k,a)],
    #   where V_star(k`) = min(Q_hat(k`, a`))
    for _ in range(episodes):
        state = 0

        while state < len(board.states) - 1:
            action = np.random.choice(len(board.dice))

            transitions = tm.transition_probabilities(board.states[state], board.dice[action], board)
            expected_value = sum(prob * (reward + gamma * np.min(Q_hat[next_state, :]))
                                 for next_state, (prob, reward) in transitions.items())
            
            Q_hat[state, action] += alpha * (expected_value - Q_hat[state, action])
            state = np.random.choice(list(transitions.keys()), p=[prob for prob, _ in transitions.values()])
    
    expectations = np.min(Q_hat, axis=1)[:-1]
    die_optimal = np.argmin(Q_hat, axis=1)[:-1] + 1  # Convert to 1-based indexing (1 for "security", 2 for "normal", 3 for "risky")
    
    return [expectations, die_optimal]


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
    expected_result = [2, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 3, 3]
    circle = True
    expectations, die_optimal = QLearningDecision(layout, circle, episodes=1000, is_test=True)
    print(expectations)
    # print(die_optimal)
    print(die_optimal[:10])
    print("     " + f'{die_optimal[10:15]}')
    print(expected_result[:10])
    print("     " + f'{expected_result[10:15]}')