import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore

df = pd.read_csv("../results/results.csv")

## Elapsed time with different strategies ##

plt.figure(figsize=(10, 6))
sns.boxplot(x='Strategy', y='Elapsed_Time', data=df)
plt.title("Elapsed Time with different strategies")
plt.xlabel('Strategy')
plt.ylabel('Elapsed Time [seconds]')
plt.xticks(rotation=45)
plt.show()


## Number of Steps (to achieve the goal) with different strategies ##

plt.figure(figsize=(10, 6))
sns.boxplot(x='Strategy', y='Steps', data=df)
plt.title("Number of Steps with different strategies")
plt.xlabel('Strategy')
plt.ylabel('Number of Steps')
plt.xticks(rotation=45)
plt.show()


## Die Distribution (for each cell) with different strategies ##

strategies_values = df['Strategy'].value_counts()

new_df = pd.DataFrame(columns=["Strategy"] + [f'Pos_{i}' for i in range(15)])
pos_dice = {}

for strat, _ in strategies_values.items():
    if strat in ['Always_Security', 'Always_Normal', 'Always_Risky', 'Risky_Then_Cautious']:
        continue
    new_df["Strategy"] = strat
    df_strat = df.loc[df['Strategy'] == strat]
    pos_dice[strat] = {}
    for i in range(15):
        frequency_dice = df_strat[f'Dice_{i}'].value_counts()
        pos_dice[strat][f'Pos_{i}'] = frequency_dice.to_dict()

bin_values = [1, 2, 3]

plt.figure(figsize=(15, 10))

for i, (strategy, positions) in enumerate(pos_dice.items()):
    
    x_vals = np.array(list(range(15)))
    y_vals = np.zeros((15, 3))
    
    for j, (pos, counts) in enumerate(positions.items()):
        pos_idx = int(pos.split('_')[1])
        for die, count in counts.items():
            die = int(die)
            if die != 0:
                y_vals[pos_idx][die - 1] += count
            
    y_vals = y_vals / y_vals.sum(axis=1, keepdims=True)
    
    plt.subplot(3, 3, i + 1)
    plt.bar(x_vals - 0.2, y_vals[:, 0], width=0.2, label='SECURITY Die', align='center')
    plt.bar(x_vals, y_vals[:, 1], width=0.2, label='NORMAL Die', align='center')
    plt.bar(x_vals + 0.2, y_vals[:, 2], width=0.2, label='RISKY Die', align='center')

    plt.title(strategy)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.xticks(x_vals)
    plt.ylim(0, 1.05)
    plt.legend(title='Die Type')

plt.tight_layout()
plt.show()