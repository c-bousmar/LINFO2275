from algorithms.QLearningDecision import QLearningDecision
from algorithms.MarkovDecision import markovDecision

if __name__ == '__main__':

    ### MDP ###
    layout = [0, 2, 3, 2, 0, 2, 2, 0, 1, 0, 0, 3, 1, 3, 0]
    circle = False
    expectations, die_optimal = markovDecision(layout, circle)
    print(expectations)
    print(die_optimal)
    
    print("\n")
    
    ### QLearning ###
    layout = [0, 2, 3, 2, 0, 2, 2, 0, 1, 0, 0, 3, 1, 3, 0]
    circle = False
    expectations, die_optimal = QLearningDecision(layout, circle, display_board=True)
    print(expectations)
    print(die_optimal)