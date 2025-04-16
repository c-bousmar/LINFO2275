#!/bin/bash
echo $PWD
MARKOV_DAT="Results/Memory/Dat/markov_memory_results.dat"
QLEARNING_DAT="Results/Memory/Dat/qlearning_memory_results.dat"
GRAPH_SCRIPT="plotter/memory_analysis.py"
ROOT="Project1/plotter/"

rm -f "$MARKOV_DAT"
rm -f "$QLEARNING_DAT"

layout="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
cycle="False"
interval="0.1"
LIMIT=50
cd "../"
for i in $(seq 1 $LIMIT); do
    echo "Iteration $i"
    echo "🔹 Profiling Markov Decision..."
    mprof run --interval $interval --include --output  $MARKOV_DAT python3 -c "import main_optimal_strategies; main_optimal_strategies.markovDecision($layout, $cycle)"
    echo "✅ Markov profiling done."
    echo "🔹 Profiling Q-Learning Decision..."
    mprof run --interval $interval --include --output $QLEARNING_DAT python3 -c "import main_optimal_strategies; main_optimal_strategies.QLearningDecision($layout, $cycle, display_board=False)"
    echo "✅ Q-Learning profiling done."
done

FILES=($MARKOV_DAT $QLEARNING_DAT)

for FILE in "${FILES[@]}"; do
    while [[ ! -f "$FILE" ]]; do
        sleep 1
    done
done

echo "🔹 Generation of the memory consumption graph..."
python3 $GRAPH_SCRIPT
echo "✅ Generation of the memory consumption graph done"