from game import TransitionManager
from game import BoardGame

from matplotlib import pyplot as plt
import numpy as np

def QLearningDecisionExperiment(layout, circle, epochs=1000, alpha=0.1, gamma=0.95, epsilon=0.1, display_board=False, stats_every=100):
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
        stats_every (int): Interval for calculating and displaying statistics.

    Returns:
        np.ndarray: The Q_hat matrix containing the estimated Q-values for each state-action pair.
        dict: Aggregated statistics (average, max, min rewards) over episodes.
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

    # Lists to store rewards and statistics
    rewards_per_epoch = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    # Q-learning algorithm
    for epoch in range(epochs):
        position = board.start_cell  # Start at the beginning of the board
        total_reward = 0

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
            
            # Accumulate reward
            total_reward += immediate_reward
            
            # Move to the next state
            if transitions:
                next_positions = list(transitions.keys())
                probabilities = [transitions[pos][0] for pos in next_positions]
                position = np.random.choice(next_positions, p=probabilities)
            else:
                break
        
        rewards_per_epoch.append(total_reward)
        
        # Decay epsilon and alpha over time for better convergence
        alpha_t = max(0.01, alpha / (1 + epoch / 2000))
        epsilon_t = max(0.01, epsilon * np.exp(-epoch / 7000))

        # Calculate statistics every `stats_every` episodes
        if not epoch % stats_every:
            average_reward = np.mean(rewards_per_epoch[-stats_every:])
            max_reward = np.max(rewards_per_epoch[-stats_every:])
            min_reward = np.min(rewards_per_epoch[-stats_every:])

            aggr_ep_rewards['ep'].append(epoch)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max_reward)
            aggr_ep_rewards['min'].append(min_reward)

            print(f'Episode: {epoch:>5d}, average reward: {average_reward:>4.1f}, max: {max_reward:>4.1f}, min: {min_reward:>4.1f}, epsilon: {epsilon_t:>1.2f}')
    
    # Compute rolling average of rewards for smoothness
    rolling_avg_rewards = np.convolve(rewards_per_epoch, np.ones(200)/200, mode='valid')

    # Extract the optimal policy and value function
    expectations = np.min(Q_hat, axis=1)[:-1]
    die_optimal = np.argmin(Q_hat, axis=1)[:-1] + 1  # Convert to 1-based indexing

    return [expectations, die_optimal], rolling_avg_rewards, aggr_ep_rewards

if __name__ == '__main__':
    # Basic Layout
    layout = [0] * 15
    circle = False

    _, rolling_avg_rewards, aggr_ep_rewards = QLearningDecisionExperiment(layout, circle, epochs=10000, alpha=0.1, gamma=0.9, epsilon=0.2, display_board=False)

    plt.figure(figsize=(12, 6))

    plt.plot(rolling_avg_rewards, label="Mean Reward (Window of 200 epoch)", color='blue')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="Moyenne (par fenêtre de 100 épisodes)", color='green')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="Max (par fenêtre de 100 épisodes)", color='red', linestyle='--')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="Min (par fenêtre de 100 épisodes)", color='purple', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("QLearning Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.show()