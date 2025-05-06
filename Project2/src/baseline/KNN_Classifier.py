import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
from distance_metrics import euclidean_distance
from collections import Counter

class KNN_Classifier:
    """
    K-Nearest Neighbors classifier for gesture recognition.
    Supports both user-independent and user-dependent classification modes.
    """
    
    def __init__(self, k=1, distance_function=None):
        """
        Initialize the KNN classifier.
        
        @param k: Number of nearest neighbors to consider
        @param distance_function: Function to compute distances between sequences
        """
        self.X_train = None
        self.y_train = None
        self.subject_info = None
        self.distance_function = distance_function or euclidean_distance
        self.k = k
        
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
    
    def fit(self, X_train, y_train, subject_info=None):
        """
        Train the KNN classifier on the provided data.
        
        @param X_train: List of training gesture sequences
        @param y_train: List of training labels
        @param subject_info: Optional list of subject IDs for user-dependent classification
        """
        # Validate input data
        if len(X_train) != len(y_train):
            raise ValueError("Number of training samples and labels must match.")
        
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty.")
        
        if subject_info is not None and len(subject_info) != len(X_train):
            raise ValueError("Number of subject IDs must match number of training samples.")

        # Store the training data
        self.X_train = X_train
        self.y_train = y_train
        self.subject_info = subject_info
        
        # Adjust k if necessary to not exceed training set size
        if self.k > len(self.X_train):
            self.k = len(self.X_train)
    
    def predict(self, X_test, verbose=True):
        """
        Predict labels for test data using user-independent approach.
        
        @param X_test: List of test gesture sequences
        @param verbose: Whether to display progress bars
        @return: List of predicted labels
        """
        # Validate model state and input
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must fit the model before predicting.")

        if len(X_test) == 0:
            raise ValueError("X_test cannot be empty.")

        predictions = []
        
        # Iterate over each test sample
        outer_iter = tqdm(X_test, desc="Predicting", unit="sample", position=0) if verbose else X_test
        for test_sample in outer_iter:
            distances = []
            
            # Compute distances to all training samples
            inner_iter = zip(self.X_train, self.y_train)
            if verbose:
                inner_iter = tqdm(inner_iter, total=len(self.X_train), desc="Comparing", position=1, leave=False)
            for train_sample, label in inner_iter:
                dist = self.distance_function(test_sample, train_sample)
                distances.append((dist, label))
            
            # Get the k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k_neighbors = [label for (_, label) in distances[:self.k]]
            
            # Determine prediction by majority vote
            most_common = Counter(k_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions