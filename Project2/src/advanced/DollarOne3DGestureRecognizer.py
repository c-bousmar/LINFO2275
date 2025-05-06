import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain

import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.spatial import procrustes


class DollarOne3DGestureRecognizer:
    """
    Implementation of the $1 recognizer algorithm adapted for 3D gesture recognition.
    Uses PCA for dimensionality reduction from 3D to 2D.
    """
    
    def __init__(self, num_points=100):
        """
        Initialize the Dollar One 3D Gesture Recognizer.
        
        @param num_points: Number of points to resample each gesture to
        """
        self.num_points = num_points
        self.pca = PCA(n_components=2)  # Reduce 3D to 2D
        self.templates = []  # Store gesture templates
    
    def _resample(self, points):
        """
        Resample the gesture points to a fixed number of equidistant points.
        
        @param points: Array of 3D points representing the gesture
        @return: Resampled points with self.num_points equidistant points
        """
        # Calculate cumulative distance along the gesture path
        cumulative_dist = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        cumulative_dist = np.insert(cumulative_dist, 0, 0)
        
        # Handle edge case of zero-length gesture
        if cumulative_dist[-1] == 0:
            return np.tile(points[0], (self.num_points, 1))
        
        # Interpolate to get equidistant points
        return interp1d(cumulative_dist, points, axis=0)(
            np.linspace(0, cumulative_dist[-1], self.num_points)
        )
    
    def _pca_project(self, points):
        """
        Project 3D points to 2D using PCA.
        
        @param points: Array of 3D points
        @return: Points projected to 2D space
        """
        return self.pca.transform(points)
    
    def _rotate_to_base(self, points):
        """
        Rotate the gesture so that the first and last points form a horizontal line.
        
        @param points: Array of 2D points
        @return: Rotated points
        """
        # Calculate angle between first and last point
        delta = points[-1] - points[0]
        angle = np.arctan2(delta[1], delta[0])
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply rotation
        return points @ rotation_matrix
    
    def _normalize_scale(self, points):
        """
        Normalize the gesture to [0,1] range on all dimensions.
        
        @param points: Array of points
        @return: Normalized points
        """
        min_val = np.min(points)
        max_val = np.max(points)
        
        # Avoid division by zero
        return (points - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else points
    
    def _preprocess(self, points):
        """
        Apply the full preprocessing pipeline to a gesture.
        
        @param points: Array of 3D points representing the gesture
        @return: Preprocessed points ready for comparison
        """
        resampled = self._resample(points)
        projected = self._pca_project(resampled)
        rotated = self._rotate_to_base(projected)
        return self._normalize_scale(rotated)

    def predict(self, X):
        """
        Predict gesture labels for a collection of gestures.
        
        @param X: List of gestures (each gesture is an array of 3D points)
        @return: Predicted labels for each gesture
        """
        predictions = []
        for gesture in X:
            pred = self.recognize(gesture)
            predictions.append(pred)
        return np.array(predictions)
    
    def fit(self, X_train, y_train):
        """
        Train the gesture recognizer by creating templates.
        
        @param X_train: List of training gestures
        @param y_train: Labels for training gestures
        """
        # Fit PCA on all resampled gestures to find best projection
        all_points = np.vstack([self._resample(g) for g in X_train])
        self.pca.fit(all_points)
        
        # Create template for each training gesture
        self.templates = []
        for gesture, label in zip(X_train, y_train):
            processed = self._preprocess(gesture)
            self.templates.append({
                'points': processed,
                'label': label
            })

    def recognize(self, gesture):
        """
        Recognize a single gesture by comparing with templates.
        
        @param gesture: Array of 3D points representing the gesture to recognize
        @return: Predicted label for the gesture
        """
        candidate = self._preprocess(gesture)
        min_score = float('inf')
        best_label = "unknown" 
        
        # Find closest matching template using Procrustes analysis
        for template in self.templates:
            _, _, disparity = procrustes(template['points'], candidate)
            if disparity < min_score:
                min_score = disparity
                best_label = template['label']
                
        return best_label


if __name__ == "__main__":
    # Load dataset
    domain_id = 1
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_id)

    # Initialize evaluator with verbose output
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate using user-independent cross-validation
    results_indep = evaluator.evaluate(
        model=DollarOne3DGestureRecognizer(),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        n_folds=10
    )
    
    # Evaluate using user-dependent cross-validation
    results_dep = evaluator.evaluate(
        model=DollarOne3DGestureRecognizer(),
        df=df,
        evaluation_type="user-dependent",
        normalize=False,
        n_folds=10
    )

    # Display results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")
    
    # Save results
    evaluator.save_results_to_csv(results_indep, f"../Results/DollarOne/_user_independent_domain{domain_id}.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/DollarOne/_user_dependent_domain{domain_id}.csv")