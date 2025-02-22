from MarkovDecision import markovDecision
from QLearningDecision import QLearningDecision

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
    
    ### MDP ###
    layout = [0, 2, 3, 2, 0, 2, 2, 0, 1, 0, 0, 3, 1, 3, 0]
    circle = False
    expectations, die_optimal = markovDecision(layout, circle)
    print(expectations)
    print(die_optimal)
    
    ### QLearning ###
    # layout = [0, 1, 3, 4, 2, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0]
    # circle = True
    # expectations, die_optimal = QLearningDecision(layout, circle, display_board=True)
    # print(expectations)
    # print(die_optimal)