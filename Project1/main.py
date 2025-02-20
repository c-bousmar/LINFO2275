from BoardGame import BoardGame

import numpy as np

def markovDecision(layout, circle):
    
    max_iterations = 1000
    tol = 1e-9
    
    game = BoardGame(layout, circle)

    expec = np.zeros(game.size, dtype=float)
    dice = np.ones(game.size, dtype=int)
    
    delta = float("inf")
    iteration = 0
    
    while (delta > tol):
        
        # TODO : Algorithm Value-Iteration
        
        # Stop the computation after too much iterations
        if (iteration == max_iterations): break;
        iteration += 1
        
        # Update delta value
        # delta = ...

    return [expec, dice]


if __name__ == '__main__':
    
    layout = np.array([0] * 15)
    circle = False

    decisions = markovDecision(layout, circle)
    
    print("Expectation: ", decisions[0])
    print("Dice: ", decisions[1])