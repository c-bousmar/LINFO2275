import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

from distance_metrics import euclidean_distance

from collections import Counter

class KNN_Classifier:
    
    def __init__(self, k=1, distance_function=None):
        self.X_train = None
        self.y_train = None
        self.subject_info = None
        self.distance_function = distance_function or euclidean_distance
        self.k = k
        
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
    
    def fit(self, X_train, y_train, subject_info=None):
        if len(X_train) != len(y_train):
            raise ValueError("Number of training samples and labels must match.")
        
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty.")
        
        if subject_info is not None and len(subject_info) != len(X_train):
            raise ValueError("Number of subject IDs must match number of training samples.")

        # Store the training points
        self.X_train = X_train
        self.y_train = y_train
        self.subject_info = subject_info
        
        if self.k > len(self.X_train):
            self.k = len(self.X_train)
    
    def predict(self, X_test, verbose=True):
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must fit the model before predicting.")

        if len(X_test) == 0:
            raise ValueError("X_test cannot be empty.")

        predictions = []
        
        # Iterate over each test sample (to predict)
        outer_iter = tqdm(X_test, desc="Predicting", unit="sample", position=0) if verbose else X_test
        for test_sample in outer_iter:
            distances = []
            
            # Iterate over each train sample (to compute distance with the test sample)
            inner_iter = zip(self.X_train, self.y_train)
            if verbose:
                inner_iter = tqdm(inner_iter, total=len(self.X_train), desc="Comparing", position=1, leave=False)
            for train_sample, label in inner_iter:
                dist = self.distance_function(test_sample, train_sample)
                distances.append((dist, label))
            
            # Get the k smallest distances
            distances.sort(key=lambda x: x[0])
            k_neighbors = [label for (_, label) in distances[:self.k]]
            
            # Make a Majority vote for the prediction
            most_common = Counter(k_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions
    
    def predict_user_dependent(self, X_test, subject_test, verbose=False):
        if self.X_train is None or self.y_train is None or self.subject_info is None:
            raise ValueError("You must fit the model with subject_info before using predict_user_dependent.")

        if len(X_test) == 0:
            raise ValueError("X_test cannot be empty.")
            
        if len(X_test) != len(subject_test):
            raise ValueError("Number of test samples and subject IDs must match.")

        predictions = []
        
        outer_iter = enumerate(zip(X_test, subject_test))
        if verbose:
            outer_iter = tqdm(outer_iter, total=len(X_test), desc="Predicting", position=0)
        for _, (test_sample, test_subject) in outer_iter:
            distances = []
            
            # Only consider training samples from the same user
            user_train_indices = [j for j, subj in enumerate(self.subject_info) if subj == test_subject]
            
            if len(user_train_indices) == 0:
                raise ValueError("There is no training samples for this user.")
            
            # Iterate over training samples from the same user
            for j in user_train_indices:
                train_sample = self.X_train[j]
                label = self.y_train[j]
                dist = self.distance_function(test_sample, train_sample)
                distances.append((dist, label))
            
            # Get the k smallest distances
            k_local = min(self.k, len(distances))
            distances.sort(key=lambda x: x[0])
            k_neighbors = [label for (_, label) in distances[:k_local]]
            
            # Make a Majority vote for the prediction
            most_common = Counter(k_neighbors).most_common(1)[0][0]
            predictions.append(most_common)
        
        return predictions
