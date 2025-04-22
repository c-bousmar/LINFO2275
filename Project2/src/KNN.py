from tqdm import tqdm

from distance_metrics import euclidean_distance

from collections import Counter

class KNN_Classifier:
    
    def __init__(self, k=1, distance_function=None):
        self.X_train = None
        self.y_train = None
        self.distance_function = distance_function or euclidean_distance
        self.k = k
        
        if self.k <= 0:
            raise ValueError("k must be a positive integer.")
    
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
    
    def predict(self, X_test, verbose=False):
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