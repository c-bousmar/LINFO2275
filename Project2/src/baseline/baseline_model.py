import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from distance_metrics import dtw_distance, edit_distance, lcs_distance
from KNN_Classifier import KNN_Classifier
from datasets_utils import get_dataset_from_domain


def load_numeric_gestures(filepath, domain_id=1):
    """Load gestures as numeric sequences (for dtw_distance)."""
    df = get_dataset_from_domain(filepath, domain_id)
    df_grouped = df.groupby("source_file")
    gestures = []
    labels = []
    for _, group in df_grouped:
        gesture = group[["<x>", "<y>", "<z>"]].values
        label = group["target"].iloc[0]
        gestures.append(gesture)
        labels.append(label)
    return gestures, labels


def load_string_gestures(filepath):
    """Load gestures as strings (for edit/lcs distance)."""
    df = pd.read_csv(filepath)
    return df["gesture"].tolist(), df["target"].tolist()


def evaluate_knn(X, y, distance_function, k=10, test_size=0.2, random_state=42):
    """Evaluate KNN classifier with given distance function."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn = KNN_Classifier(k=k, distance_function=distance_function)
    print("Training KNN...")
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test, verbose=True)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y))
    return acc, cm, y_test, predictions


def plot_confusion_matrix(cm, labels, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (KNN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', format="pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN on gesture sequences.")
    parser.add_argument("--input", type=str, required=True, help="Path to the gesture dataset CSV.")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for KNN.")
    parser.add_argument("--distance", choices=["dtw", "edit", "lcs"], default="lcs",
                        help="Distance metric to use: 'dtw', 'edit', or 'lcs' (default: lcs).")
    parser.add_argument("--domain", type=int, default=1, help="Domain ID of the dataset (only for DTW).")
    parser.add_argument("--output", type=str, default="baseline/results/confusion_matrix.pdf", help="Output image path.")
    args = parser.parse_args()

    # Select distance function based on argument
    if args.distance == "dtw":
        distance_fn = dtw_distance
        print("Using dtw distance.")
        # Load numeric gestures for dtw
        print("Loading numeric gesture data...")
        X, y = load_numeric_gestures(args.input, domain_id=args.domain)
    elif args.distance == "edit":
        distance_fn = edit_distance
        print("Using edit distance.")
        # Load string gestures for edit distance
        print("Loading string gesture data...")
        X, y = load_string_gestures(args.input)
    elif args.distance == "lcs":
        distance_fn = lcs_distance
        print("Using normalized LCS distance.")
        # Load string gestures for LCS distance
        print("Loading string gesture data...")
        X, y = load_string_gestures(args.input)
    else:
        print("Unknown distance")
        sys.exit(1)

    # Evaluate the KNN classifier
    acc, cm, _, _ = evaluate_knn(X, y, distance_function=distance_fn, k=args.k)

    print(f"Accuracy: {acc:.4f}")
    print("Saving confusion matrix...")
    plot_confusion_matrix(cm, labels=np.unique(y), output_path=args.output)


if __name__ == "__main__":
    main()