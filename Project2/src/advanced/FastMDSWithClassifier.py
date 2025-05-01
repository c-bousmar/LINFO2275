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
    performance improvements to speed up the computation.
    """
    
    def __init__(self, 
                 classifier_type='logistic', 
                 distance_function=dtw_distance,
                 n_components=5,
                 use_features=True,
                 use_caching=True,
                 sample_ratio=0.05,  # Reduced to 5% for much faster computation
                 fast_dtw=True,
                 n_jobs=-1,
                 verbose=True):
        """
        Initialize the model.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier to use ('logistic', 'svm', or 'nn')
        distance_function : function
            Function to compute distances between sequences
        n_components : int
            Number of dimensions for MDS representation
        use_features : bool
            Whether to augment MDS with statistical features
        use_caching : bool
            Whether to cache distance computations
        sample_ratio : float
            Ratio of sequences to sample (between 0 and 1)
        fast_dtw : bool
            Whether to use a faster DTW implementation
        n_jobs : int
            Number of parallel jobs (-1 for all CPUs)
        verbose : bool
            Whether to display progress
        """
        self.distance_function = distance_function
        self.n_components = n_components
        self.use_features = use_features
        self.use_caching = use_caching
        self.sample_ratio = sample_ratio
        self.fast_dtw = fast_dtw
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Create classifier pipeline
        if classifier_type == 'logistic':
            self.classifier = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, C=1, multi_class='multinomial'))
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
        
        # Initialize attributes
        self.mds_model = None
        self.train_sequences = None
        self.sampled_train_sequences = None
        self.sample_indices = None
        self.feature_scaler = None
        self.mds_features = None  # Store MDS features explicitly
        self.cache_file = "mds_distance_cache.joblib"
        
        # For fast DTW if enabled
        if self.fast_dtw:
            try:
                from fastdtw import fastdtw # type: ignore
                self.fastdtw = fastdtw
            except ImportError:
                if self.verbose:
                    print("fastdtw package not found. Install with: pip install fastdtw")
                self.fast_dtw = False
    
    def _sample_sequences(self, sequences, labels):
        """
        Sample a subset of sequences to reduce computation time.
        
        Parameters:
        -----------
        sequences : list
            List of sequences
        labels : list
            List of labels
            
        Returns:
        --------
        tuple
            (sampled_sequences, sampled_labels, indices)
        """
        if self.sample_ratio >= 1.0:
            return sequences, labels, np.arange(len(sequences))
        
        n = len(sequences)

        # Stratified sampling by label
        unique_labels = np.unique(labels)
        indices = []
        
        for label in unique_labels:
            label_indices = np.where(np.array(labels) == label)[0]
            label_sample_size = max(int(len(label_indices) * self.sample_ratio), 1)
            sampled_indices = np.random.choice(label_indices, size=label_sample_size, replace=False)
            indices.extend(sampled_indices)
        
        # Convert to numpy array and sort
        indices = np.array(indices)
        indices.sort()
        
        sampled_sequences = [sequences[i] for i in indices]
        sampled_labels = [labels[i] for i in indices]
        
        if self.verbose:
            print(f"Sampled {len(indices)} sequences from {n} ({self.sample_ratio:.1%})")
        
        return sampled_sequences, sampled_labels, indices
    
    def _compute_dtw_distance_matrix_parallel(self, sequences):
        """
        Compute DTW distance matrix using parallel processing.
        
        Parameters:
        -----------
        sequences : list
            List of sequences
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix
        """
        n = len(sequences)
        distances = np.zeros((n, n))
        
        # Check for cache
        if self.use_caching and os.path.exists(self.cache_file):
            if self.verbose:
                print(f"Loading cached distances from {self.cache_file}...")
            try:
                distances = load(self.cache_file)
                if distances.shape == (n, n):
                    return distances
                else:
                    if self.verbose:
                        print(f"Cache shape {distances.shape} doesn't match required shape {(n, n)}. Recomputing...")
            except:
                if self.verbose:
                    print("Error loading cache. Recomputing distances...")
        
        # Function to compute a single distance
        def compute_distance(i, j):
            if i <= j:  # Only compute upper triangle
                if self.fast_dtw:
                    dist, _ = self.fastdtw(sequences[i], sequences[j])
                    return i, j, dist
                else:
                    return i, j, self.distance_function(sequences[i], sequences[j])
            return i, j, 0  # Will be filled in by symmetry
        
        # Generate all pairs
        pairs = [(i, j) for i in range(n) for j in range(i, n)]
        
        # Compute distances in parallel
        start_time = time.time()
        results = Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
            delayed(compute_distance)(i, j) for i, j in pairs
        )
        
        # Fill the distance matrix
        for i, j, dist in results:
            distances[i, j] = dist
            if i != j:  # Fill the lower triangle by symmetry
                distances[j, i] = dist
        
        if self.verbose:
            print(f"Distance computation took {time.time() - start_time:.2f} seconds")
        
        # Cache the results
        if self.use_caching:
            if self.verbose:
                print(f"Caching distances to {self.cache_file}...")
            dump(distances, self.cache_file)
        
        return distances
    
    def _extract_features(self, sequences):
        """
        Extract statistical features from sequences.
        
        Parameters:
        -----------
        sequences : list
            List of sequences
            
        Returns:
        --------
        numpy.ndarray
            Feature matrix
        """
        if self.verbose:
            print("Extracting statistical features...")
        
        features = np.array([extract_features_from_gesture(seq) for seq in sequences])
        return features
    
    def fit(self, X_train, y_train):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X_train : list
            List of gesture sequences
        y_train : list
            List of labels
        """
        if self.verbose:
            print(f"Fitting FastMDS model on {len(X_train)} sequences...")
        
        # Store the original training sequences
        self.train_sequences = X_train
        
        # Sample sequences for MDS computation
        self.sampled_train_sequences, sampled_y_train, self.sample_indices = self._sample_sequences(X_train, y_train)
        
        # Compute distance matrix for sampled sequences
        if self.verbose:
            print(f"Computing distance matrix for {len(self.sampled_train_sequences)} sampled sequences...")
        distance_matrix = self._compute_dtw_distance_matrix_parallel(self.sampled_train_sequences)
        
        # Apply MDS to sampled sequences
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
        
        # Extract statistical features for all training sequences
        if self.use_features:
            # Extract statistical features
            statistical_features = self._extract_features(X_train)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            statistical_features = self.feature_scaler.fit_transform(statistical_features)
            
            # For each sequence, either use its MDS features if it was in the sample,
            # or approximate them using the nearest sampled sequence
            full_mds_features = np.zeros((len(X_train), self.n_components))
            
            # For sampled sequences, use their direct MDS features
            for i, orig_idx in enumerate(self.sample_indices):
                full_mds_features[orig_idx] = self.mds_features[i]
            
            # For non-sampled sequences, approximate using the nearest sampled one
            non_sampled_indices = np.setdiff1d(np.arange(len(X_train)), self.sample_indices)
            
            if len(non_sampled_indices) > 0:
                if self.verbose:
                    print(f"Approximating MDS features for {len(non_sampled_indices)} non-sampled sequences...")
                
                for idx in non_sampled_indices:
                    # Find the nearest sampled sequence using statistical features
                    dists = np.sum((statistical_features[self.sample_indices] - statistical_features[idx])**2, axis=1)
                    nearest_idx = np.argmin(dists)
                    # Use its MDS features
                    full_mds_features[idx] = self.mds_features[nearest_idx]
            
            # Combine MDS and statistical features
            combined_features = np.hstack([full_mds_features, statistical_features])
            training_features = combined_features
        else:
            # Only MDS features (only for sampled sequences)
            training_features = self.mds_features
            y_train = sampled_y_train
        
        # Train classifier
        if self.verbose:
            print("Training classifier...")
        self.classifier.fit(training_features, y_train)
    
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        Parameters:
        -----------
        X_test : list
            List of test sequences
            
        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        if self.mds_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute distances between test sequences and sampled training sequences
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
                # Use inverse distance as weights - make sure dimensions match
                weights = 1.0 / (test_distance_matrix[i] + 1e-10)  # Avoid division by zero
                weights /= np.sum(weights)  # Normalize
                mds_test_features[i, j] = np.sum(weights * self.mds_features[:, j])
        
        # Prepare features for prediction
        if self.use_features:
            # Extract and scale statistical features
            test_statistical_features = self._extract_features(X_test)
            test_statistical_features = self.feature_scaler.transform(test_statistical_features)
            
            # Combine MDS and statistical features
            test_features = np.hstack([mds_test_features, test_statistical_features])
        else:
            test_features = mds_test_features
        
        # Make predictions
        if self.verbose:
            print("Predicting labels...")
        y_pred = self.classifier.predict(test_features)
        
        return y_pred


if __name__ == "__main__":
    # Get dataset
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)
    
    # Create evaluator
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate MDS model
    results_indep = evaluator.evaluate(
        model=FastMDSWithClassifier(
            classifier_type='logistic',
            distance_function=dtw_distance,
            n_components=5,
            use_features=True,
            use_caching=True,         # Cache distances to avoid recomputation
            sample_ratio=0.05,        # Use 5% of the data for extreme speed
            fast_dtw=True,            # Use faster DTW implementation
            n_jobs=-1,
            verbose=True
        ),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    results_dep = evaluator.evaluate(
        model=FastMDSWithClassifier(
            classifier_type='logistic',
            distance_function=dtw_distance,
            n_components=5,
            use_features=True,
            use_caching=True,         # Cache distances to avoid recomputation
            sample_ratio=0.05,        # Use 5% of the data for extreme speed
            fast_dtw=True,            # Use faster DTW implementation
            n_jobs=-1,
            verbose=True
        ),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )
    
    # Print results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Independent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")