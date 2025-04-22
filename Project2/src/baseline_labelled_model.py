import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from distance_metrics import edit_distance, lcs_distance
from KNN import KNN_Classifier


def load_labelled_gestures(filepath):
    df = pd.read_csv(filepath)
    return df["gesture"].tolist(), df["target"].tolist()


def evaluate_knn(X, y, distance_function, k=10, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn = KNN_Classifier(k=k, distance_function=distance_function)
    print("Training KNN...")
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test, verbose=True)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y))
    return acc, cm, y_test, predictions


def plot_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (KNN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN on labelled gesture sequences.")
    parser.add_argument("--input", type=str, required=True, help="Path to the labelled gesture CSV.")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for KNN.")
    parser.add_argument("--distance", choices=["edit", "lcs"], default="lcs",
                        help="Distance metric to use: 'edit' or 'lcs' (default: lcs).")
    parser.add_argument("--output", type=str, default="../results/confusion_matrix.png", help="Output image path.")
    args = parser.parse_args()

    if args.distance == "edit":
        distance_fn = edit_distance
        print("Using edit distance.")
    elif args.distance == "lcs":
        distance_fn = lcs_distance
        print("Using normalized LCS distance.")
    else:
        print("Unknown distance")
        exit(1)

    print("Loading data...")
    X, y = load_labelled_gestures(args.input)

    acc, cm, y_test, y_pred = evaluate_knn(X, y, distance_function=distance_fn, k=args.k)

    print(f"Accuracy: {acc:.4f}")
    print("Saving confusion matrix...")
    plot_confusion_matrix(cm, labels=np.unique(y), output_path=args.output)


if __name__ == "__main__":
    main()
