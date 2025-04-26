import sys
import os
sys.path.append(os.path.abspath('../'))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datasets_utils import get_dataset_from_domain
from sklearn.model_selection import train_test_split
from KNN_Classifier import KNN_Classifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

#######################################################
### To Remove (Duplication inside distance_metrics.py)
###

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous[j + 1] + 1
            deletions = current[j] + 1
            substitutions = previous[j] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def load_data(file_path):
    df = get_dataset_from_domain(file_path, domain_number=1)
    df = df.sample(frac=0.5, random_state=42)
    grouped = df.groupby("source_file")
    gestures = []
    labels = []

    for name, group in grouped:
        gesture = group[["<x>", "<y>", "<z>"]].values
        label = group["target"].iloc[0]
        gestures.append(gesture)
        labels.append(label)
    return np.array(gestures, dtype=object), np.array(labels)


def gestures_to_strings(gestures, scaler, kmeans):
    strings = []
    for gesture in gestures:
        gesture_norm = scaler.transform(gesture)
        labels = kmeans.predict(gesture_norm)
        strings.append(''.join([chr(65 + label) for label in labels]))
    return strings


def gesture_classification(X, y, n_clusters=2):
    all_points = np.vstack(X)

    scaler = StandardScaler().fit(all_points)
    all_points_norm = scaler.transform(all_points)

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=42).fit(all_points_norm)

    X_strings = gestures_to_strings(X, scaler, kmeans)
    X_train, X_test, y_train, y_test = train_test_split(
        X_strings, y, test_size=0.2, random_state=42)
    knn_dtw = KNN_Classifier(k=10, distance_function=edit_distance)
    knn_dtw.fit(X_train, y_train)

    predictions = knn_dtw.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Confusion Matrix (ED-KNN)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    X, y = load_data("../Data/dataset.csv")

    h_params = {
        'n_clusters': 6,
    }
    gesture_classification(X, y, **h_params)
