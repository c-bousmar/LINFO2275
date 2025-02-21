from BoardGame import BoardGame
from DiceStrategy import DiceStrategy
from LayoutGenerator import LayoutGenerator
from BoardGameSimulator import BoardGameSimulator
from markovDecision import markovDecision

layout_properties = {"Layout 1" : [0.0, 0.1, 0.05, 0.4],
                    #  "Layout 2" : [0.0, 0.3, 0.0, 0.0],
                    #  "Layout 3" : [0.0, 0.0, 0.0, 0.3],
                    #  "Layout 4" : [0.0, 0.1, 0.3, 0.0],
}

if __name__ == "__main__":
    
    for layout_name, event_density in layout_properties.items():
        generator = LayoutGenerator(event_density)
        layout = generator.generate_layout()
        circle = False
        board = BoardGame(layout, circle)
        board.display_board()
        
        # print(markovDecision(layout, circle)[1])
        dice_strategy = DiceStrategy(layout, circle)
        game_simulator = BoardGameSimulator(board, dice_strategy, n_simulations=10)
        game_simulator.compare_strategies()