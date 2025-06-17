from game.LayoutGenerator import LayoutGenerator
from game.BoardGame import BoardGame

from simulator.BoardGameSimulator import BoardGameSimulator
from simulator.DiceStrategy import DiceStrategy

layouts_properties = {"Layout 1" : [0.0, 0.1, 0.05, 0.4],
                      "Layout 2" : [0.0, 0.3, 0.0, 0.0],
                      "Layout 3" : [0.0, 0.0, 0.0, 0.3],
                      "Layout 4" : [0.0, 0.1, 0.3, 0.0],
}

layouts = [[1]*15]

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
        game_simulator = BoardGameSimulator(board, dice_strategy, n_simulations=10000)
        game_simulator.compare_strategies()