from Game.TransitionManager import TransitionManager

import numpy as np
import time
import os
import csv

class BoardGameSimulator:
    
    def __init__(self, board, dice_strategies, n_simulations=1000, save_path="Results/"):
        self.board = board 
        self.tm = TransitionManager(board)
        self.dice_strategies = dice_strategies
        self.n_simulations = n_simulations
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.csv_file = os.path.join(self.save_path, "simulations.csv")
        
        if not os.path.exists(self.csv_file):
            columns = ["Strategy", "Steps", "Elapsed_Time"] + \
                        [f"Dice_{i}" for i in range(15)] + \
                        [f"Exp_{i}" for i in range(15)] + \
                        [f"Layout_{i}" for i in range(15)]
            with open(self.csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(columns)

    def compare_strategies(self):
        for strategy_name, strategy in self.dice_strategies.strategies.items():
            print(f"Running simulations for strategy: {strategy_name}")
            self.run_simulations(strategy, strategy_name)
            print(f"Completed: {strategy_name}\n")
    
    def run_simulations(self, strategy, strategy_name):
        all_expectations = []  
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for _ in range(self.n_simulations):
                start = time.time()
                nb_steps, dice_policy, expectations = self.simulate_game(strategy)
                end = time.time()
                elapsed_time = end - start
                all_expectations.append(expectations)
                row = [strategy_name, nb_steps, elapsed_time] + dice_policy + expectations + list(self.board.layout)
                writer.writerow(row)
                file.flush()
        avg_expectations = np.mean(all_expectations, axis=0)
        print("Moyenne des expectations pour chaque position :", avg_expectations)
    
    def get_cost_to_goal_state(self, position):
        self.board.fast_lane
        if position in self.board.fast_lane:
            return self.board.last_cell - position
        else:
            return self.board.slow_lane.stop - position
                
    def simulate_game(self, strategy):
        dice_sums = np.zeros(15)
        dice_counts = np.zeros(15)
        exp_sums = np.zeros(15)
        exp_counts = np.zeros(15)
        
        state = self.board.states[0]
        steps = 0
        while state.position != self.board.last_cell:
            die = strategy(state.position)       
            move = die.roll()
            
            dice_sums[state.position] += die.type.value
            dice_counts[state.position] += 1
            exp_sums[state.position] += self.get_cost_to_goal_state(state.position)
            exp_counts[state.position] += 1
            
            offsets = [0, 8] if (state.position + 1 == self.board.slow_lane.start) else [0]
            offset = np.random.choice(offsets)
            new_position = self.tm.get_next_position(state.position, move, offset)
            new_state = self.board.states[new_position]
            if die.is_triggering_event():
                new_position, extra_cost = self.tm.handling_events(new_state)
                steps += extra_cost

            state = self.board.states[new_position]
            steps += 1
            
        mask = dice_counts > 0
        
        dice_policy = np.zeros_like(dice_sums)
        expectations = np.zeros_like(exp_sums)

        dice_policy[mask] = dice_sums[mask] / dice_counts[mask]
        expectations[mask] = exp_sums[mask] / exp_counts[mask]

        dice_policy = dice_policy.tolist()
        expectations = expectations.tolist()

        return steps, dice_policy, expectations