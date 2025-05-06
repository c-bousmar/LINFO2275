import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from joblib import Parallel, delayed, dump, load
import os.path

from sklearn.manifold import MDS
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from datasets_utils import get_dataset_from_domain, extract_features_from_gesture
from baseline.distance_metrics import dtw_distance
from GestureRecognizerEstimator import GestureRecognitionEvaluator


class FastMDSWithClassifier:
    """
    An optimized model that uses Multi-Dimensional Scaling (MDS) with various
    performance improvements for fast gesture recognition.
    """
    
    def __init__(self, 
                 classifier_type='logistic',
                 distance_function=dtw_distance,
                 n_components=5,
                 use_caching=True,
                 sample_ratio=0.05,
                 fast_dtw=True,
                 n_jobs=-1,
                 verbose=True):
        """
        Initialize the FastMDS with Classifier model.
        
        @param classifier_type: Type of classifier to use ('logistic', 'svm', or 'nn')
        @param distance_function: Function to compute distances between sequences
        @param n_components: Number of dimensions for MDS representation
        @param use_caching: Whether to cache distance computations
        @param sample_ratio: Ratio of sequences to sample (between 0 and 1)
        @param fast_dtw: Whether to use a faster DTW implementation
        @param n_jobs: Number of parallel jobs (-1 for all CPUs)
        @param verbose: Whether to display progress
        """
        self.distance_function = distance_function
        self.n_components = n_components
        self.use_caching = use_caching
        self.sample_ratio = sample_ratio
        self.fast_dtw = fast_dtw
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Create classifier pipeline based on the specified type
        if classifier_type == 'logistic':
            self.classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, C=1.0, penalty='l2', solver='liblinear'))
            ])
        elif classifier_type == 'svm':
            self.classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(C=10, gamma='scale', probability=True))
            ])
        elif classifier_type == 'nn':
            self.classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True))
            ])
        else:
            raise ValueError("classifier_type must be 'logistic', 'svm', or 'nn'")
        
        # Initialize model attributes
        self.mds_model = None
        self.train_sequences = None
        self.sampled_train_sequences = None
        self.sample_indices = None
        self.feature_scaler = None
        self.mds_features = None
        self.cache_file = "mds_distance_cache.joblib"
        
        # Import fast DTW if enabled
        if self.fast_dtw:
            from fastdtw import fastdtw
            self.fastdtw = fastdtw
    
    def _sample_sequences(self, sequences, labels):
        """
        Sample a subset of sequences to reduce computation time.
        
        @param sequences: List of sequences
        @param labels: List of labels
        @return: Tuple (sampled_sequences, sampled_labels, indices)
        """
        # If sample ratio is 1 or greater, use all sequences
        if self.sample_ratio >= 1.0:
            return sequences, labels, np.arange(len(sequences))
        
        n = len(sequences)

        # Perform stratified sampling by label to maintain class distribution
        unique_labels = np.unique(labels)
        indices = []
        
        for label in unique_labels:
            # Get indices of samples with this label
            label_indices = np.where(np.array(labels) == label)[0]
            # Calculate how many to sample from this class
            label_sample_size = max(int(len(label_indices) * self.sample_ratio), 1)
            # Sample without replacement
            sampled_indices = np.random.choice(label_indices, size=label_sample_size, replace=False)
            indices.extend(sampled_indices)
        
        # Convert to numpy array and sort for consistent access
        indices = np.array(indices)
        indices.sort()
        
        # Create the sampled dataset
        sampled_sequences = [sequences[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
        
        if self.verbose:
            print(f"Sampled {len(indices)} sequences from {n} ({self.sample_ratio:.1%})")
        
        return sampled_sequences, sampled_labels, indices
    
    def _compute_dtw_distance_matrix_parallel(self, sequences):
        """
        Compute DTW distance matrix using parallel processing.
        
        @param sequences: List of sequences
        @return: Distance matrix as numpy array
        """
        n = len(sequences)
        distances = np.zeros((n, n))
        
        self.cache_file = f"mds_distance_cache_{n}.joblib"
        
        if self.use_caching and os.path.exists(self.cache_file):
            if self.verbose:
                print(f"Loading cached distances from {self.cache_file}...")
            try:
                cached_distances = load(self.cache_file)
                if cached_distances.shape == (n, n):
                    return cached_distances
                else:
                    if self.verbose:
                        print(f"Cache shape {cached_distances.shape} doesn't match required shape {(n, n)}. Recomputing...")
            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache: {e}. Recomputing distances...")
        
        # Function to compute a single distance
        def compute_distance(i, j):
            if i <= j:  # Only compute upper triangle (matrix is symmetric)
                if self.fast_dtw:
                    dist, _ = self.fastdtw(sequences[i], sequences[j])
                    return i, j, dist
                else:
                    return i, j, self.distance_function(sequences[i], sequences[j])
            return i, j, 0  # Will be filled in by symmetry
        
        # Generate all pairs to compute
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        
        # Compute distances in parallel
        start_time = time.time()
        results = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
            delayed(compute_distance)(i, j) for i, j in pairs
        )
        
        # Fill the distance matrix with results
        for i, j, dist in results:
            distances[i, j] = dist
            if i != j:  # Fill the lower triangle by symmetry
                distances[j, i] = dist
        
        if self.verbose:
            print(f"Distance computation took {time.time() - start_time:.2f} seconds")
        
        # Cache the computed distances
        if self.use_caching:
            if self.verbose:
                print(f"Caching distances to {self.cache_file}...")
            dump(distances, self.cache_file)
        
        return distances
    
    def _extract_features(self, sequences):
        """
        Extract statistical features from sequences.
        
        @param sequences: List of sequences
        @return: Feature matrix as numpy array
        """
        if self.verbose:
            print("Extracting statistical features...")
        
        features = np.array([extract_features_from_gesture(seq) for seq in sequences])
        return features
    
    def fit(self, X_train, y_train):
        """
        Fit the model to training data.
        
        @param X_train: List of training gesture sequences
        @param y_train: List of training labels
        """
        if self.verbose:
            print(f"Fitting FastMDS model on {len(X_train)} sequences...")
        
        # Store the original training sequences
        self.train_sequences = X_train
        
        # Sample sequences to reduce computation for MDS
        self.sampled_train_sequences, sampled_y_train, self.sample_indices = self._sample_sequences(X_train, y_train)
        
        # Compute distance matrix for sampled sequences
        if self.verbose:
            print(f"Computing distance matrix for {len(self.sampled_train_sequences)} sampled sequences...")
        distance_matrix = self._compute_dtw_distance_matrix_parallel(self.sampled_train_sequences)
        
        # Apply MDS to learn lower-dimensional representation
        if self.verbose:
            print(f"Applying MDS to learn {self.n_components}-dimensional representation...")
        self.mds_model = MDS(
            n_components=self.n_components,
            dissimilarity='precomputed',
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=2 if self.verbose else 0
        )
        self.mds_features = self.mds_model.fit_transform(distance_matrix)
        training_features = self.mds_features
        y_train = sampled_y_train
        
        # Train classifier on the prepared features
        if self.verbose:
            print("Training classifier...")
        self.classifier.fit(training_features, y_train)
    
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        @param X_test: List of test gesture sequences
        @return: Predicted labels as numpy array
        """
        if self.mds_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute distances between test and sampled training sequences
        if self.verbose:
            print(f"Computing distances between {len(X_test)} test and {len(self.sampled_train_sequences)} sampled training sequences...")
        
        n_test = len(X_test)
        n_train_sampled = len(self.sampled_train_sequences)
        
        # Function to compute distances for one test sample
        def compute_test_distances(test_idx):
            test_seq = X_test[test_idx]
            distances = np.zeros(n_train_sampled)
            
            for train_idx in range(n_train_sampled):
                if self.fast_dtw:
                    dist, _ = self.fastdtw(test_seq, self.sampled_train_sequences[train_idx])
                    distances[train_idx] = dist
                else:
                    distances[train_idx] = self.distance_function(test_seq, self.sampled_train_sequences[train_idx])
            
            return distances
        
        # Compute all test distances in parallel
        start_time = time.time()
        test_distances = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
            delayed(compute_test_distances)(i) for i in range(n_test)
        )
        test_distance_matrix = np.array(test_distances)
        
        if self.verbose:
            print(f"Test distance computation took {time.time() - start_time:.2f} seconds")
        
        # Project test sequences into MDS space
        if self.verbose:
            print("Projecting test sequences...")
        
        mds_test_features = np.zeros((n_test, self.n_components))
        
        # Use weighted average based on inverse distances for projection
        for i in range(n_test):
            for j in range(self.n_components):
                # Use inverse distance as weights
                weights = 1.0 / (test_distance_matrix[i] + 1e-10)  # Avoid division by zero
                weights /= np.sum(weights)  # Normalize weights
                mds_test_features[i, j] = np.sum(weights * self.mds_features[:, j])
        
        test_features = mds_test_features
        
        # Make predictions using the trained classifier
        if self.verbose:
            print("Predicting labels...")
        y_pred = self.classifier.predict(test_features)
        
        return y_pred


if __name__ == '__main__':
    # Load dataset
    domain_id = 1
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_id)

    # Initialize evaluator with verbose output
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate using user-independent cross-validation
    results_indep = evaluator.evaluate(
        model=FastMDSWithClassifier(
            classifier_type='logistic',
            distance_function=dtw_distance,
            n_components=20,
            use_caching=True,         # Cache distances to avoid recomputation
            sample_ratio=0.4,         # Use 40% of the data for extreme speed
            fast_dtw=True,            # Use faster DTW implementation
            n_jobs=-1,
            verbose=False
        ),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    # Evaluate using user-dependent cross-validation
    results_dep = evaluator.evaluate(
        model=FastMDSWithClassifier(
            classifier_type='logistic',
            distance_function=dtw_distance,
            n_components=20,
            use_caching=True,         # Cache distances to avoid recomputation
            sample_ratio=0.4,         # Use 40% of the data for extreme speed
            fast_dtw=True,            # Use faster DTW implementation
            n_jobs=-1,
            verbose=False
        ),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )

    # Display results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")
    
    # Save results
    evaluator.save_results_to_csv(results_indep, f"../Results/FastMDS/_user_independent_domain{domain_id}.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/FastMDS/_user_dependent_domain{domain_id}.csv")