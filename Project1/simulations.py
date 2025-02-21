from BoardGame import BoardGame
from LayoutGenerator import LayoutGenerator
from BoardGameSimulator import BoardGameSimulator

layout_properties = {"Layout 1" : [0.2, 0.1, 0.1, 0.3],
                     "Layout 2" : [0.0, 0.3, 0.0, 0.0],
                     "Layout 3" : [0.0, 0.0, 0.0, 0.3],
                     "Layout 4" : [0.0, 0.1, 0.3, 0.0],
}

if __name__ == "__main__":
    
    for layout_name, event_density in layout_properties.items():
        generator = LayoutGenerator(event_density)
        layout = generator.generate_layout()
        circle = False
        board = BoardGame(layout, circle)
        simulator = BoardGameSimulator(board, n_simulations=10)