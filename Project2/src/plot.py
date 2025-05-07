import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.style.use('seaborn-v0_8-whitegrid')

cmap_confusion = LinearSegmentedColormap.from_list("confusion_cmap", ["#f7fbff", "#08306b"])
cmap_diff = LinearSegmentedColormap.from_list("diff_cmap", ["#d73027", "#f7f7f7", "#1a9850"])

domain1_labels = [str(i) for i in range(10)]

domain1_dep_data = np.array([
    [98, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 98, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 99, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 100, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 99, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 97, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 100, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
])

domain1_indep_data = np.array([
    [96, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    [0, 94, 6, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 93, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 98, 0, 2, 0, 0, 0, 0],
    [1, 0, 0, 0, 99, 0, 0, 0, 0, 0],
    [0, 2, 1, 3, 1, 93, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 92, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
    [2, 0, 0, 0, 0, 0, 5, 0, 93, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 98]
])

domain4_labels = ["Cone", "Cuboid", "Cylinder", "CylPipe", "Hemisphere", 
                  "Pyramid", "RectPipe", "Sphere", "Tetrahedron", "Toroid"]

domain4_dep_data = np.array([
    [98, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 99, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 97, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 99, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 94, 0, 0, 6, 0],
    [0, 1, 0, 1, 0, 0, 98, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 96, 0, 2],
    [3, 0, 0, 0, 0, 9, 1, 0, 87, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 100]
])

domain4_indep_data = np.array([
    [91, 0, 0, 0, 1, 2, 1, 0, 5, 0],
    [0, 99, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 97, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 2, 82, 0, 0, 10, 6, 0, 0],
    [0, 0, 1, 1, 96, 1, 0, 0, 0, 1],
    [6, 0, 0, 0, 0, 86, 0, 0, 8, 0],
    [1, 0, 0, 3, 0, 0, 96, 0, 0, 0],
    [0, 0, 0, 10, 0, 0, 0, 84, 0, 6],
    [9, 1, 0, 1, 0, 23, 0, 0, 66, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 97]
])

def calculate_metrics(dep_matrix, indep_matrix):
    dep_acc = np.diag(dep_matrix) / np.sum(dep_matrix, axis=1) * 100
    indep_acc = np.diag(indep_matrix) / np.sum(indep_matrix, axis=1) * 100
    
    diff = dep_acc - indep_acc
    
    dep_global = np.sum(np.diag(dep_matrix)) / np.sum(dep_matrix) * 100
    indep_global = np.sum(np.diag(indep_matrix)) / np.sum(indep_matrix) * 100
    
    return {
        'dep_acc': dep_acc,
        'indep_acc': indep_acc,
        'diff': diff,
        'dep_global': dep_global,
        'indep_global': indep_global
    }

def find_top_confusions(matrix, labels, n=5):
    confusions = []
    
    matrix_copy = matrix.copy()
    np.fill_diagonal(matrix_copy, 0)
    
    for i in range(len(labels)):
        row_sum = np.sum(matrix[i])
        for j in range(len(labels)):
            if i != j and matrix[i][j] > 0:
                confusions.append({
                    'from': labels[i],
                    'to': labels[j],
                    'count': matrix[i][j],
                    'percentage': (matrix[i][j] / row_sum) * 100
                })
    
    confusions.sort(key=lambda x: x['count'], reverse=True)
    return confusions[:n]

def plot_confusion_matrix(matrix, labels, title, normalize=True, colorbar_label="Pourcentage (%)"):
    plt.figure(figsize=(10, 8))
    
    if normalize:
        matrix_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis] * 100
        fmt = '.1f'
    else:
        matrix_norm = matrix
        fmt = 'd'
    
    ax = sns.heatmap(matrix_norm, annot=True, fmt=fmt, cmap=cmap_confusion,
                      xticklabels=labels, yticklabels=labels, vmin=0, 
                      vmax=100 if normalize else None, cbar_kws={'label': colorbar_label})
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('Vraie classe', fontsize=14)
    plt.xlabel('Classe prédite', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    plt.tight_layout()
    return plt

def plot_comparison_accuracies(metrics, labels, title, sort_by='diff'):
    _, ax = plt.subplots(figsize=(12, 6))
    
    indices = np.arange(len(labels))
    width = 0.35
    
    if sort_by == 'diff':
        sort_indices = np.argsort(metrics['diff'])[::-1]
        sorted_labels = [labels[i] for i in sort_indices]
        dep_acc = metrics['dep_acc'][sort_indices]
        indep_acc = metrics['indep_acc'][sort_indices]
    else:
        sorted_labels = labels
        dep_acc = metrics['dep_acc']
        indep_acc = metrics['indep_acc']
    
    bars1 = ax.bar(indices - width/2, dep_acc, width, label='User-dependent', color='#4285F4')
    bars2 = ax.bar(indices + width/2, indep_acc, width, label='User-independent', color='#DB4437')
    
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Précision (%)', fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xticks(indices)
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
    ax.legend()
    
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_differences(metrics, labels, title, threshold=5):
    sort_indices = np.argsort(metrics['diff'])[::-1]
    sorted_labels = [labels[i] for i in sort_indices]
    sorted_diff = metrics['diff'][sort_indices]
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(np.arange(len(sorted_labels)), sorted_diff, 
                  color=[('#D32F2F' if d > threshold else '#1976D2') for d in sorted_diff])
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Différence de précision (%)', fontsize=14)
    ax.set_xticks(np.arange(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')
    
    ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, 
              label=f'Seuil de sensibilité ({threshold}%)')
    
    ax.text(len(sorted_labels)-1, threshold+1, 'Forte sensibilité à l\'utilisateur', 
            ha='right', color='#D32F2F', fontweight='bold')
    ax.text(len(sorted_labels)-1, threshold-2, 'Faible sensibilité à l\'utilisateur', 
            ha='right', color='#1976D2', fontweight='bold')
    
    plt.tight_layout()
    return plt

def plot_top_confusions(confusions, title):
    labels = [f"{conf['from']} → {conf['to']}" for conf in confusions]
    values = [conf['percentage'] for conf in confusions]
    counts = [conf['count'] for conf in confusions]
    
    sort_indices = np.argsort(values)
    labels = [labels[i] for i in sort_indices]
    values = [values[i] for i in sort_indices]
    counts = [counts[i] for i in sort_indices]
    
    colors = []
    for val in values:
        if val > 20:
            colors.append('#D50000')
        elif val > 10:
            colors.append('#FF6D00')
        elif val > 5:
            colors.append('#FFD600')
        else:
            colors.append('#AEEA00')
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(np.arange(len(labels)), values, color=colors)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}% ({count})', va='center')
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Pourcentage de confusion (%)', fontsize=14)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    ax.axvline(x=5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=20, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_confusion_network(
    matrix: np.ndarray,
    labels: list[str],
    title: str = "Confusion Network",
    cmap_name: str = 'copper',
    node_color: str = '#66b3ff',
    figsize: tuple[float, float] = (8, 6),
    seed: int = 42
) -> plt.Figure:
    
    G = nx.DiGraph()

    for i, label in enumerate(labels):
        total = matrix[i].sum()
        accuracy = (matrix[i, i] / total * 100) if total > 0 else 0.0
        G.add_node(label, accuracy=accuracy)

    for i, src in enumerate(labels):
        row_sum = matrix[i].sum()
        if row_sum == 0:
            continue
        for j, dst in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                w = matrix[i, j] / row_sum * 100
                G.add_edge(src, dst, weight=w)

    pos = nx.spring_layout(G, k=2.0, scale=0.5, seed=seed)

    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color=node_color,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )

    node_labels = {n: f"{n}\n{G.nodes[n]['accuracy']:.1f}%" for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=10,
        font_weight='bold',
        ax=ax
    )

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if weights:
        norm = Normalize(vmin=0.0, vmax=max(weights))
    else:
        norm = Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(cmap_name)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    nx.draw_networkx_edges(
        G, pos,
        edge_color=weights,
        edge_cmap=cmap,
        edge_vmin=norm.vmin,
        edge_vmax=norm.vmax,
        width=[2 + (w - norm.vmin) / (norm.vmax - norm.vmin) * 4 for w in weights],
        arrowstyle='-|>',
        arrowsize=20,
        connectionstyle='arc3,rad=0.2',
        min_source_margin=15,
        min_target_margin=15,
        ax=ax
    )

    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confusion Weight (%)', rotation=270, labelpad=15)

    ax.set_title(title, fontsize=20, weight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    return fig

metrics_domain1 = calculate_metrics(domain1_dep_data, domain1_indep_data)
metrics_domain4 = calculate_metrics(domain4_dep_data, domain4_indep_data)

confusions_domain1 = find_top_confusions(domain1_indep_data, domain1_labels, n=5)
confusions_domain4 = find_top_confusions(domain4_indep_data, domain4_labels, n=5)

def generate_all_visualizations():
    output_folder = './'
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    cm_domain1 = plot_confusion_matrix(
        domain1_indep_data, domain1_labels, 
        'Matrice de confusion - DenseNet User-Independent - Chiffres (0-9)'
    )
    cm_domain1.savefig(f'{output_folder}confusion_matrix_domain1.png', dpi=300, bbox_inches='tight')
    
    cm_domain4 = plot_confusion_matrix(
        domain4_indep_data, domain4_labels, 
        'Matrice de confusion - DenseNet User-Independent - Formes 3D'
    )
    cm_domain4.savefig(f'{output_folder}confusion_matrix_domain4.png', dpi=300, bbox_inches='tight')
    
    comp_domain1 = plot_comparison_accuracies(
        metrics_domain1, domain1_labels, 
        'Comparaison des précisions - DenseNet - Chiffres (0-9)'
    )
    comp_domain1.savefig(f'{output_folder}accuracy_comparison_domain1.png', dpi=300, bbox_inches='tight')
    
    comp_domain4 = plot_comparison_accuracies(
        metrics_domain4, domain4_labels, 
        'Comparaison des précisions - DenseNet - Formes 3D'
    )
    comp_domain4.savefig(f'{output_folder}accuracy_comparison_domain4.png', dpi=300, bbox_inches='tight')
    
    diff_domain1 = plot_differences(
        metrics_domain1, domain1_labels, 
        'Sensibilité à l\'utilisateur - DenseNet - Chiffres (0-9)',
        threshold=5
    )
    diff_domain1.savefig(f'{output_folder}accuracy_differences_domain1.png', dpi=300, bbox_inches='tight')
    
    diff_domain4 = plot_differences(
        metrics_domain4, domain4_labels, 
        'Sensibilité à l\'utilisateur - DenseNet - Formes 3D',
        threshold=5
    )
    diff_domain4.savefig(f'{output_folder}accuracy_differences_domain4.png', dpi=300, bbox_inches='tight')
    
    conf_domain1 = plot_top_confusions(
        confusions_domain1, 
        'Principales confusions - DenseNet User-Independent - Chiffres (0-9)'
    )
    conf_domain1.savefig(f'{output_folder}top_confusions_domain1.png', dpi=300, bbox_inches='tight')
    
    conf_domain4 = plot_top_confusions(
        confusions_domain4, 
        'Principales confusions - DenseNet User-Independent - Formes 3D'
    )
    conf_domain4.savefig(f'{output_folder}top_confusions_domain4.png', dpi=300, bbox_inches='tight')
    
    net_domain1 = plot_confusion_network(
        domain1_indep_data, domain1_labels, 
        'Confusion Network - User-Independent - Digits (0-9)'
    )
    net_domain1.savefig(f'{output_folder}confusion_network_domain1_indep.png', dpi=300, bbox_inches='tight')
    
    net_domain1 = plot_confusion_network(
        domain1_dep_data, domain1_labels, 
        'Confusion Network - User-Dependent - Digits (0-9)'
    )
    net_domain1.savefig(f'{output_folder}confusion_network_domain1_dep.png', dpi=300, bbox_inches='tight')
    
    net_domain4 = plot_confusion_network(
        domain4_indep_data, domain4_labels, 
        'Confusion Network - User-Independent - 3D Shapes'
    )
    net_domain4.savefig(f'{output_folder}confusion_network_domain4_indep.png', dpi=300, bbox_inches='tight')
    
    net_domain4 = plot_confusion_network(
        domain4_dep_data, domain4_labels, 
        'Confusion Network - DenseNet User-Dependent - 3D Shapes'
    )
    net_domain4.savefig(f'{output_folder}confusion_network_domain4_dep.png', dpi=300, bbox_inches='tight')
    
    print("=== Résumé d'analyse des performances du DenseNet ===")
    print("\nDomain1 (Chiffres 0-9):")
    print(f"  User-dependent:   Précision globale = {metrics_domain1['dep_global']:.1f}%")
    print(f"  User-independent: Précision globale = {metrics_domain1['indep_global']:.1f}%")
    print(f"  Différence: {metrics_domain1['dep_global'] - metrics_domain1['indep_global']:.1f}%")
    
    print("\nDomain4 (Formes 3D):")
    print(f"  User-dependent:   Précision globale = {metrics_domain4['dep_global']:.1f}%")
    print(f"  User-independent: Précision globale = {metrics_domain4['indep_global']:.1f}%")
    print(f"  Différence: {metrics_domain4['dep_global'] - metrics_domain4['indep_global']:.1f}%")
    
    print("\nTop 5 des confusions - Domain1 (User-independent):")
    for i, conf in enumerate(confusions_domain1):
        print(f"  {i+1}. {conf['from']}->{conf['to']}: {conf['count']} occurrences ({conf['percentage']:.1f}%)")
    
    print("\nTop 5 des confusions - Domain4 (User-independent):")
    for i, conf in enumerate(confusions_domain4):
        print(f"  {i+1}. {conf['from']}->{conf['to']}: {conf['count']} occurrences ({conf['percentage']:.1f}%)")
    
    print("\nClasses les plus difficiles à classifier (User-independent):")
    domain1_hardest = np.argsort(metrics_domain1['indep_acc'])[:3]
    domain4_hardest = np.argsort(metrics_domain4['indep_acc'])[:3]
    
    print("  Domain1:", ", ".join([f"{domain1_labels[i]} ({metrics_domain1['indep_acc'][i]:.1f}%)" for i in domain1_hardest]))
    print("  Domain4:", ", ".join([f"{domain4_labels[i]} ({metrics_domain4['indep_acc'][i]:.1f}%)" for i in domain4_hardest]))
    
    print("\nClasses les plus sensibles à l'utilisateur:")
    domain1_sensitive = np.argsort(metrics_domain1['diff'])[::-1][:3]
    domain4_sensitive = np.argsort(metrics_domain4['diff'])[::-1][:3]
    
    print("  Domain1:", ", ".join([f"{domain1_labels[i]} ({metrics_domain1['diff'][i]:.1f}%)" for i in domain1_sensitive]))
    print("  Domain4:", ", ".join([f"{domain4_labels[i]} ({metrics_domain4['diff'][i]:.1f}%)" for i in domain4_sensitive]))
    
    print("\nVisualisation terminée ! Toutes les figures ont été enregistrées dans le dossier:", output_folder)

if __name__ == "__main__":
    generate_all_visualizations()