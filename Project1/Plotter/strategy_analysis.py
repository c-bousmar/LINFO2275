from algorithms import markovDecision

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def heatmap_strategy_expect(filename):
    """
    Generates a heatmap that visualizes the expectation values across different strategies.
    
    Parameters:
    filename (str): The path to the CSV file containing the simulation data.
    
    The function reads the simulation data, extracts the expectation values for each strategy,
    and creates a heatmap where the rows represent the strategies and the columns represent the
    positions on the board.
    """
    data = pd.read_csv(filename)
    strategies = data['Strategy'].unique()
    
    expectation_matrix  = pd.DataFrame(columns=[f'Pos_{i}' for i in range(14)])

    for strategy in strategies:
        strategy_data = data[data['Strategy'] == strategy]
        expectation_matrix.loc[strategy] = [strategy_data[f'Expectation_{i}'].values[0] for i in range(14)]

    plt.figure(figsize=(14, 10))
    sns.heatmap(expectation_matrix, annot=True, cmap='Blues', cbar=True, linewidths=0.5, fmt='.2f')
    plt.title("Expectation Values Across Different Strategies", fontsize=20, fontweight='bold')
    plt.xlabel("Position on Board", fontsize=14)
    plt.ylabel("Strategies", fontsize=14)
    plt.tight_layout()
    plt.show()
    
def lineplot_strategy_expect(filename):
    """
    Generates a refined line plot comparing the evolution of expectations over positions for each strategy.
    
    Parameters:
    filename (str): The path to the CSV file containing the simulation data.
    
    The function reads the simulation data, extracts the expectation values for each strategy,
    compares them with the theoretical expectations from the Markov Decision Process (MDP),
    and visualizes the data using an enhanced line plot.
    """
    data = pd.read_csv(filename)
    data['Strategy'] = data['Strategy'].str.replace('Optimal_', '', regex=True)
    strategies = data['Strategy'].unique()
    
    layout_columns = [f'Layout_Pos_{i}' for i in range(15)]
    layout = data[layout_columns].iloc[0].tolist()
    circle = data["Circle"][0]
    
    expect_theoretical = markovDecision(layout, circle)[0]
    expectation_matrix = pd.DataFrame(columns=['Strategy'] + [f'Pos_{i}' for i in range(14)])
    
    for strategy in strategies:
        strategy_data = data[data['Strategy'] == strategy]
        row = {'Strategy': strategy}
        row.update({f'Pos_{i}': strategy_data[f'Expectation_{i}'].values[0] for i in range(14)})
        expectation_matrix = pd.concat([expectation_matrix, pd.DataFrame([row])], ignore_index=True)
    
    result_dict = {'Strategy': 'MDP-Theoretical'}
    for i, value in enumerate(expect_theoretical):
        result_dict[f'Pos_{i}'] = np.float64(value)
    
    expectation_matrix = pd.concat([expectation_matrix, pd.DataFrame([result_dict])], ignore_index=True)
    expectation_melted = expectation_matrix.melt(id_vars=['Strategy'], var_name='Position', value_name='Expectation')
    
    expectation_melted['Position'] = expectation_melted['Position'].str.extract('(\d+)').astype(int) + 1
    
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10", len(strategies) + 1)
    
    ax = sns.lineplot(
        data=expectation_melted, x='Position', y='Expectation', 
        hue='Strategy', marker="o", linewidth=2.5, palette=palette
    )

    
    plt.xlabel("Board Position", fontsize=18, fontweight='bold')
    plt.ylabel("Expectation Value", fontsize=18, fontweight='bold')
    plt.xticks(range(1, 15), fontsize=14)
    plt.yticks(fontsize=14)
    
    for line in ax.lines:
        if line.get_label() == 'MDP-Theoretical':
            line.set_linestyle("dashed")
            line.set_linewidth(3)
            line.set_alpha(0.8)
    
    legend = plt.legend(title="Strategies", title_fontsize=18, fontsize=16, loc='upper right')
    legend.get_frame().set_alpha(0.9)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Generate the plots
# heatmap_strategy_expect("Results/simulations.csv")
lineplot_strategy_expect("../Results/simu.csv")