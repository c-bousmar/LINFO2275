from Game.TransitionManager import TransitionManager
from Game.BoardGame import BoardGame

import numpy as np

def QLearningDecision(layout, circle, epochs=10000, alpha=0.1, gamma=0.95, epsilon=0.1, display_board=False):
    """
    Implements Q-learning to estimate the optimal policy with ε-greedy exploration.
    This algorithm learns the Q-values for state-action pairs through interaction with the environment.

    Args:
        layout (list): The layout of the board, specifying the type of each cell.
        circle (bool): Indicates whether the board is circular.
        alpha (float): The learning rate, controlling how quickly the algorithm updates Q-values.
        gamma (float): The discount factor, balancing immediate and future rewards.
        epsilon (float): The exploration rate, determining the probability of taking random actions.
        epochs (int): The number of learning episodes (iterations) to run.
        display_board (bool): Whether to display the board during the learning process.

    Returns:
        np.ndarray: The Q_hat matrix containing the estimated Q-values for each state-action pair.
    """
    # Initialize board and transition manager
    board = BoardGame(layout, circle)
    tm = TransitionManager(board)
    
    alpha_t = alpha
    epsilon_t = epsilon
    
    if display_board: 
        board.display_board()

    # Initialize Q-values
    Q_hat = np.zeros((len(board.states), len(board.dice)), dtype=float)

    # Q-learning algorithm
    for epoch in range(epochs):
        position = board.start_cell  # Start at the beginning of the board

        while position < board.last_cell:
            # Exploration vs Exploitation: ε-greedy action selection
            if np.random.rand() < epsilon_t:  # Exploration
                die_id = np.random.choice(len(board.dice))
            else:  # Exploitation: choose the die with the highest Q-value
                die_id = np.argmax(Q_hat[position, :])
            
            # Get the chosen die and its transitions
            die = board.dice[die_id]
            transitions = tm.transition_probabilities(board.states[position], die)
            
            # Calculate expected immediate reward and future value
            immediate_reward = 1.0
            future_value = sum(
                extra + p * np.max(Q_hat[next_pos, :])  
                for next_pos, (p, extra) in transitions.items()
            )

            # Update Q-value using the Q-learning formula
            Q_hat[position, die_id] += alpha_t * (
                immediate_reward + gamma * future_value - Q_hat[position, die_id]
            )
            
            # Move to the next state
            if transitions:
                next_positions = list(transitions.keys())
                probabilities = [transitions[pos][0] for pos in next_positions]
                position = np.random.choice(next_positions, p=probabilities)
            else:
                break
        
        alpha_t = max(0.01, alpha / (1 + epoch / 2000))
        epsilon_t = max(0.01, epsilon * np.exp(-epoch / 7000))
    
    # Extract the optimal policy and value function
    expectations = np.min(Q_hat, axis=1)[:-1]
    die_optimal = np.argmin(Q_hat, axis=1)[:-1] + 1 # Convert to 1-based indexing
    
    return [expectations, die_optimal]