from markovDecision import get_cost_to_goal_state

import numpy as np
import time
import os
import csv

from Enum import CellType, PositionType

class BoardGameSimulator:
    def __init__(self, board, dice_strategies, n_simulations=1000, save_path="results/"):
        self.board = board
        self.dice_strategies = dice_strategies
        self.n_simulations = n_simulations
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.csv_file = os.path.join(self.save_path, "simulations.csv")
        
        if not os.path.exists(self.csv_file):
            columns = ["Strategy", "Steps", "Elapsed_Time"] + \
                        [f"Layout_{i}" for i in range(15)] + \
                        [f"Dice_{i}" for i in range(15)] + \
                        [f"Exp_{i}" for i in range(15)]
            with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(columns)
    
    def simulate_game(self, strategy):
        state = self.board.states[0]
        
        dice_sums = np.zeros(15)
        dice_counts = np.zeros(15)

        exp_sums = np.zeros(15)
        exp_counts = np.zeros(15)
        
        steps = 0
        while state.position != PositionType.FINAL_CELL.value:
            die_choice = strategy(state.position)
            
            dice_sums[state.position] += die_choice
            dice_counts[state.position] += 1
            exp_sums[state.position] += get_cost_to_goal_state(state.position)
            exp_counts[state.position] += 1
            
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

        dice_policy = list(np.where(dice_counts > 0, dice_sums / dice_counts, 0))
        expectations = list(np.where(exp_counts > 0, exp_sums / exp_counts, 0))

        return steps, dice_policy, expectations
    
    def run_simulations(self, strategy, strategy_name):
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for _ in range(self.n_simulations):
                start = time.time()
                nb_steps, dice_policy, expectations = self.simulate_game(strategy)
                end = time.time()
                elapsed_time = end - start
    
                row = [strategy_name, nb_steps, elapsed_time] + list(self.board.layout) + dice_policy + expectations 
                writer.writerow(row)
                file.flush()
    
    def compare_strategies(self):
        for strategy_name, strategy in self.dice_strategies.strategies.items():
            print(f"Running simulations for strategy: {strategy_name}")
            self.run_simulations(strategy, strategy_name)
            print(f"Completed: {strategy_name}\n")