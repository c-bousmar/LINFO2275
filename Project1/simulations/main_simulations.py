import sys
sys.path.append('/Users/mathisdelsart/Desktop/Github-Repository/LINFO2275/Project1')

from BoardGameSimulator import BoardGameSimulator
from DiceStrategy import DiceStrategy

from Game.LayoutGenerator import LayoutGenerator
from Game.BoardGame import BoardGame

layouts_properties = {"Layout 1" : [0.0, 0.1, 0.05, 0.4],
                      "Layout 2" : [0.0, 0.3, 0.0, 0.0],
                      "Layout 3" : [0.0, 0.0, 0.0, 0.3],
                      "Layout 4" : [0.0, 0.1, 0.3, 0.0],
}

layouts = [[0, 2, 3, 2, 0, 2, 2, 0, 1, 0, 0, 3, 1, 3, 0]]

strategy_names = ["Optimal_MDP"]

if __name__ == "__main__":
    
    if layouts == []:
        for layout_name, event_density in layouts_properties.items():
            generator = LayoutGenerator(event_density)
            layout = generator.generate_layout()
            layouts.append((layout_name, layout))
            
    for layout in layouts:
        circle = False
        board = BoardGame(layout, circle)
        dice_strategy = DiceStrategy(board, strategy_names)
        board.display_board()
        game_simulator = BoardGameSimulator(board, dice_strategy, n_simulations=10)
        game_simulator.compare_strategies()