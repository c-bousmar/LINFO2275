import numpy as np
import pandas as pd
import os

from Enum import CellType, PositionType, DieType
from BoardGame import BoardGame
from markovDecision import markovDecision

class BoardGameSimulator:
    def __init__(self, board, n_simulations=1000, save_path="results/"):
        self.board = board
        self.n_simulations = n_simulations
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
    
    def simulate_game(self, strategy):
        state = self.board.states[0]
        steps = 0
        while state.position != PositionType.FINAL_CELL.value:
            die_choice = strategy(state.position)
            die = next(d for d in self.board.dice if d.type.value == die_choice)
            move = die.roll()
            new_position = min(state.position + move, PositionType.FINAL_CELL.value)
            new_state = self.board.states[new_position]
            if new_state.cell_type == CellType.RESTART and die.is_triggering_event():
                new_position = 0
            elif new_state.cell_type == CellType.PENALTY and die.is_triggering_event():
                new_position = max(0, new_position - 3)
            elif new_state.cell_type == CellType.PRISON and die.is_triggering_event():
                steps += 1
            elif new_state.cell_type == CellType.BONUS and die.is_triggering_event():
                steps = -1    
            state = self.board.states[new_position]
            steps += 1
        return steps
    
    def run_simulations(self, strategy, strategy_name):
        results = []
        for _ in range(self.n_simulations):
            steps = self.simulate_game(strategy)
            results.append(steps)
        
        print(f"Average steps: {np.mean(results)}")
        print(f"Standard deviation: {np.std(results)}")
        print(f"Median steps: {np.median(results)}")
        
        df = pd.DataFrame(results, columns=["Steps"])
        df.to_csv(os.path.join(self.save_path, f"{strategy_name}.csv"), index=False)
        print(f"Simulation results saved to {strategy_name}.csv")
    
    def compare_strategies(self, strategies):
        for name, strategy in strategies.items():
            print(f"Running simulations for strategy: {name}")
            self.run_simulations(strategy, name)

def always_choose_security(position):
    return DieType.SECURITY.value

def always_choose_normal(position):
    return DieType.NORMAL.value

def always_choose_risky(position):
    return DieType.RISKY.value

def random_strategy(position):
    return np.random.choice([DieType.SECURITY.value, DieType.NORMAL.value, DieType.RISKY.value])

def optimal_strategy(position):
    return optimal_policy[position]

if __name__ == "__main__":
    
    layout = np.array([0] * 15)
    board = BoardGame(layout, circle=False)
    
    optimal_policy = markovDecision(board)[1]
    
    simulator = BoardGameSimulator(board, n_simulations=10000)
    
    strategies = {
        "Always Security": always_choose_security,
        "Always Normal": always_choose_normal,
        "Always Risky": always_choose_risky,
        "Random": random_strategy,
        "Optimal": optimal_strategy
    }
    
    # simulator.compare_strategies(strategies)
    simulator.run_simulations(strategies["Always Normal"], "Always Normal")