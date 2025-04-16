from collections import Counter
import numpy as np

class KNN_Classifier:
    
    def __init__(self, k=1, distance_func=None):
        self.X_train = None
        self.y_train = None
        self.distance_func = distance_func or self.euclidean_distance
        self.k = k
        
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
    
    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
        
    def fit(self, X_train, y_train):
        if len(X_train) != len(y_train):
            raise ValueError("Number of training samples and labels must match.")
        
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty.")

        # Store the training points
        self.X_train = X_train
        self.y_train = y_train
        
        if self.k > len(self.X_train):
            self.k = len(self.X_train)
    
    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must fit the model before predicting.")

        if len(X_test) == 0:
            raise ValueError("X_test cannot be empty.")

        predictions = []
        
        # Iterate over each test sample (to predict)
        for test_sample in X_test:
            distances = []
            
            # Iterate over each train sample (to compute distance with the test sample)
            for train_sample, train_sample_label in zip(self.X_train, self.y_train):
                dist = self.distance_func(test_sample, train_sample)
                distances.append((dist, train_sample_label))
            
            # Get the k smallest distances
            distances.sort(key=lambda x: x[0])
            k_neighbors = [label for (_, label) in distances[:self.k]]
            
            # Make a Majority vote for the prediction
            most_common = Counter(k_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions
    
def visualize_example(k=1):
    
    import matplotlib.pyplot as plt

    # Training Points
    X_train = np.array([
        # Class 0 (6 points)
        [1, 2], [2, 3], [3, 4],
        [4, 2], [2, 1], [3, 2],
        # Class 1 (7 points)
        [6, 6], [7, 7], [8, 8],
        [5, 7], [6, 8], [7, 4], [10, 10]
    ])

    y_train = [
        # Class 0 (6 points)
        0, 0, 0, 0, 0, 0,
        # Class 1 (7 points)
        1, 1, 1, 1, 1, 1, 1
    ]

    # Test Points
    X_test = np.array([[2, 2], [5, 5], [4, 5], [5, 4], [7, 6]])

    # Fitting the model and predict
    KNN = KNN_Classifier(k=k)
    KNN.fit(X_train, y_train)
    predictions = KNN.predict(X_test)
    print("Predictions:", predictions)

    # Visualization
    colors = ['blue', 'red']
    labels = ['Class 0', 'Class 1']

    for label in np.unique(y_train):
        plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f"Train: {labels[label]}", s=100, edgecolors='k')

    shown_labels = set()
    for i, point in enumerate(X_test):
        pred_label = predictions[i]
        if pred_label not in shown_labels:
            plt.scatter(point[0], point[1], marker='X', color=colors[pred_label], s=200, label=f"Test: Predict {labels[pred_label]}")
            shown_labels.add(pred_label)
        else:
            plt.scatter(point[0], point[1], marker='X', color=colors[pred_label], s=200)

    plt.legend()
    plt.title("KNN Classifier - Visualization")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.show()
    
visualize_example(k=3)