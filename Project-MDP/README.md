# Optimal Strategy for Snakes and Ladders using MDP and Q-Learning

## Project Overview

This project presents a computational analysis of a stochastic version of the classic game *Snakes and Ladders*, using Markov Decision Processes (MDP) and Reinforcement Learning (Q-learning). The objective is to determine optimal strategies for dice selection in order to minimize the expected number of moves required to reach the final square.

This work was completed as part of the course **LINFO2275 – Data Mining and Decision Making** at UCLouvain.

---

## Objective

To compute, for every position on the board, the optimal dice choice (Security, Normal, or Risky) that minimizes the expected number of moves to reach the goal square. The analysis considers two game variants:

- **Exact End:** The player must land exactly on the final square to win.
- **Flexible End:** The player wins upon reaching or surpassing the final square.

---

## Game Description

The game board consists of 15 positions, starting from square 1 and ending at square 15.

### Dice Types

- **Security Die:** Moves the player 0 or 1 cell (equal probability); immune to cell effects.
- **Normal Die:** Moves 0–2 cells (equal probability); 50% chance to trigger cell effect.
- **Risky Die:** Moves 0–3 cells (equal probability); always triggers cell effect.

### Cell Effects

Each board position may have a cell effect:

- **Restart (1):** Resets the player to square 1.
- **Penalty (2):** Moves the player back 3 positions.
- **Prison (3):** Player loses one turn.
- **Bonus (4):** Grants the player an additional turn.

---

## Methodology

### Markov Decision Process (MDP)

An MDP is defined as a tuple (S, A, P, C, γ), where:

- **S:** Set of states (board positions)
- **A:** Set of actions (dice choices)
- **P:** Transition probability function
- **C:** Immediate cost (fixed to 1 per move)
- **γ:** Discount factor, set to 1 (episodic task)
- **π:** Policy mapping states to actions
- **V:** Value function representing the expected number of moves from each state


### Q-Learning

Q-learning is implemented as a model-free learning algorithm with a specific update rule.

---

## Implementation Details

### `markovDecision(layout, circle)`

- **Input:**
  - `layout`: A NumPy array representing the board configuration.
  - `circle`: A boolean indicating whether the game requires an exact landing.

- **Output:**
  - `Expec`: Expected number of moves from each position.
  - `Dice`: Optimal dice choice for each position.

Additional strategies are evaluated for comparison:

- AlwaysSecurity
- AlwaysNormal
- AlwaysRisky
- Random
- Optimal via MDP
- Optimal via Q-Learning

---

## Evaluation and Results

### Test Configurations

Several board layouts were tested, including:

- Layouts with only one type of trap (Bonus, Penalty, Prison, Restart)
- A mixed layout including all trap types

### Performance Metrics

- Expected number of moves per strategy
- Execution time and memory usage
- Dice selection frequency
- State visitation heatmaps

---

## Authors

- Cyril Bousmar – SINF – cyril.bousmar@uclouvain.be  
- Mathis Delsart – INFO – mathis.delsart@student.uclouvain.be  
- Sienou Lamien – SINF – sienou.lamien@student.uclouvain.be