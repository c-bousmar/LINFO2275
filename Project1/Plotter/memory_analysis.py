import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import psutil
import os

markov_dat = "Results/Memory/Dat/markov_memory_results.dat"
markov_csv = "Results/Memory/Csv/markov_memory_results.csv"

qlearning_dat = "Results/Memory/Dat/qlearning_memory_results.dat"
qlearning_csv = "Results/Memory/Csv/qlearning_memory_results.csv"

markov_graph = "Results/Memory/Markov/markov_memory_custom.png"
qlearning_graph = "Results/Memory/Qlearning/qlearning_memory_custom.png"

qlearning_markov_graph = "Results/Memory/QlearningMarkov/memory_comparison.png"
qlearning_markov_graph_log_scale = "Results/Memory/QlearningMarkov/memory_comparison_log_scale.png"

def is_run_from_bash():
    """Check if the script is running from a Bash shell."""
    parent = psutil.Process(os.getppid()).name()
    return parent in ["bash", "sh", "zsh", "fish"]

if not is_run_from_bash():
    os.chdir("Project1")

def get_file_path(file_name):
    """Return the absolute file path."""
    return os.path.realpath(file_name)

def convert_mprof_to_csv(dat_file, csv_file):
    """Convert mprof .dat file to a .csv file."""
    with open(dat_file, "r") as f:
        lines = f.readlines()

    data = {}
    interval = 0.1
    simulation = 0

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            memory, _ = map(float, parts[1:])
            if simulation in data:
                data[simulation].append(memory)
            else:
                data[simulation] = [memory]
            simulation += 1
        else:
            simulation = 0

    df = pd.DataFrame(columns=["SimulationID", "Mean Memory (MB)", "Std Memory (MB)", "Time (s)"])
    for index, element in data.items():
        df.loc[len(df)] = [index+1, np.mean(element), np.std(element), index*interval]

    df.to_csv(csv_file, index=False)

def plot_error_bars(time, mean, std, label, color, marker, linestyle, capsize=5, markersize=8):
    """Plot the error bars for memory data."""
    plt.errorbar(time, mean, yerr=std, label=label,
                 color=color, marker=marker, capsize=capsize, 
                 linestyle=linestyle, markersize=markersize)

def plot_memory_comparison1(markov_csv, qlearning_csv, output_graph):

    markov_data = pd.read_csv(markov_csv)
    qlearning_data = pd.read_csv(qlearning_csv)

    markov_time = markov_data["Time (s)"]
    markov_mean = markov_data["Mean Memory (MB)"]
    markov_std = markov_data["Std Memory (MB)"]

    qlearning_time = qlearning_data["Time (s)"]
    qlearning_mean = qlearning_data["Mean Memory (MB)"]
    qlearning_std = qlearning_data["Std Memory (MB)"]

    plt.figure(figsize=(10, 5))

    plt.errorbar(markov_time, markov_mean, yerr=markov_std, label="Markov Decision",
                 color="blue", marker="o", capsize=5, linestyle="-")

    plt.errorbar(qlearning_time[::4], qlearning_mean[::4], yerr=qlearning_std[::4], label="QLearning",
                 color="red", marker="o", capsize=5, linestyle="-")

    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.yscale("log")
    plt.title("Memory Usage Comparison: Markov vs QLearning")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_graph)


def plot_memory_comparison2(markov_csv, qlearning_csv, output_graph):
    markov_data = pd.read_csv(markov_csv)
    qlearning_data = pd.read_csv(qlearning_csv)

    markov_time = markov_data["Time (s)"]
    markov_mean = markov_data["Mean Memory (MB)"]
    markov_std = markov_data["Std Memory (MB)"]

    qlearning_time = qlearning_data["Time (s)"]
    qlearning_mean = qlearning_data["Mean Memory (MB)"]
    qlearning_std = qlearning_data["Std Memory (MB)"]

    plt.figure(figsize=(12, 6))
    min_time = min(markov_time.min(), qlearning_time.min())
    shift = 0.01 if min_time == 0 else 0
    plt.errorbar(markov_time + shift, markov_mean, yerr=markov_std, label="Markov Decision",
                 color="blue", marker="o", capsize=4, linestyle="-", markersize=8, linewidth=2,
                 alpha=0.8, markerfacecolor="lightblue", markeredgewidth=1.5)

    plt.errorbar(qlearning_time + shift, qlearning_mean, yerr=qlearning_std, label="QLearning",
                 color="red", marker="s", capsize=4, linestyle="-", markersize=8, linewidth=2,
                 alpha=0.8, markerfacecolor="lightcoral", markeredgewidth=1.5)

    plt.xscale("symlog", linthresh=0.1)
    plt.yscale("symlog", linthresh=0.1)
    x_ticks = np.concatenate([np.arange(0, 1, 0.1), np.arange(
        1, max(markov_time.max(), qlearning_time.max()), 1)])
    plt.xticks(x_ticks, rotation=45)

    plt.yticks([1, 10, 100, 1000])

    plt.xlabel("Time (s)", fontsize=14, fontweight="bold")
    plt.ylabel("Memory (MB)", fontsize=14, fontweight="bold")
    plt.title("Memory Usage Comparison: Markov vs QLearning",
              fontsize=16, fontweight="bold")

    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    plt.legend(fontsize=12, loc="upper left",
               bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(output_graph, dpi=300)

def plot_memory(filename, title, output_image):
    """Plot the memory usage from a CSV file."""
    df = pd.read_csv(filename)
    plt.figure(figsize=(10, 5))
    plt.errorbar(df["Time (s)"], df["Mean Memory (MB)"], yerr=df["Std Memory (MB)"], label="Memory usage", color="blue", marker="o", capsize=5, linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory (MB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_image)

# Convert .dat files to .csv
convert_mprof_to_csv(get_file_path(markov_dat), markov_csv)
convert_mprof_to_csv(get_file_path(qlearning_dat), qlearning_csv)

# Plot individual memory usage graphs
plot_memory(get_file_path(markov_csv), "Memory profiling - Markov Decision", get_file_path(markov_graph))
plot_memory(get_file_path(qlearning_csv), "Memory profiling - QLearning", get_file_path(qlearning_graph))

# Plot comparison graphs
plot_memory_comparison1(markov_csv, qlearning_csv, get_file_path(qlearning_markov_graph))
plot_memory_comparison2(markov_csv, qlearning_csv, get_file_path(qlearning_markov_graph_log_scale))