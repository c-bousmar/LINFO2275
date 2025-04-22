from datasets_utils import get_dataset_from_domain
from distance_metrics import edit_distance, DTW, euclidean_distance, lcs_distance, lcs_length

from KNN import KNN_Classifier

import numpy as np

df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)
df = df.sample(frac=0.5, random_state=42)

grouped = df.groupby("source_file")

gestures = []
labels = []

for name, group in grouped:
    gesture = group[["<x>", "<y>", "<z>"]].values
    label = group["target"].iloc[0]
    gestures.append(gesture)
    labels.append(label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(gestures, labels, test_size=0.2, random_state=42)

knn_dtw = KNN_Classifier(k=10, distance_function=lcs_length)
knn_dtw.fit(X_train, y_train)

predictions = knn_dtw.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

acc = accuracy_score(y_test, predictions)
print(f"Accuracy: {acc:.2f}")

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(labels), yticklabels=np.unique(labels))
plt.title("Confusion Matrix (DTW-KNN)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", bbox_inches='tight')
plt.show()
