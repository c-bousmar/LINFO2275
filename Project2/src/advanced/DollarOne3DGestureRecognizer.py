import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain

import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.spatial import procrustes
from scipy.interpolate import interp1d


class DollarOne3DGestureRecognizer:
    
    def __init__(self, num_points=100):
        self.num_points = num_points
        self.pca = PCA(n_components=2)
        self.templates = []
    
    def _resample(self, points):
        cumulative_dist = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        cumulative_dist = np.insert(cumulative_dist, 0, 0)
        if cumulative_dist[-1] == 0:
            return np.tile(points[0], (self.num_points, 1))
        
        return interp1d(cumulative_dist, points, axis=0)(np.linspace(0, cumulative_dist[-1], self.num_points))
    
    def _pca_project(self, points):
        return self.pca.transform(points)
    
    def _rotate_to_base(self, points):
        delta = points[-1] - points[0]
        angle = np.arctan2(delta[1], delta[0])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix
    
    def _normalize_scale(self, points):
        min_val = np.min(points)
        max_val = np.max(points)
        return (points - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else points
    
    def _preprocess(self, points):
        resampled = self._resample(points)
        projected = self._pca_project(resampled)
        rotated = self._rotate_to_base(projected)
        return self._normalize_scale(rotated)

    def predict(self, X):
        predictions = []
        for gesture in X:
            pred = self.recognize(gesture)
            predictions.append(pred)
        return np.array(predictions)
    
    def fit(self, X_train, y_train):
        all_points = np.vstack([self._resample(g) for g in X_train])
        self.pca.fit(all_points)
        
        self.templates = []
        for gesture, label in zip(X_train, y_train):
            processed = self._preprocess(gesture)
            self.templates.append({
                'points': processed,
                'label': label
            })

    def recognize(self, gesture):
        candidate = self._preprocess(gesture)
        min_score = float('inf')
        best_label = "unknown" 
        for template in self.templates:
            _, _, disparity = procrustes(template['points'], candidate)
            if disparity < min_score:
                min_score = disparity
                best_label = template['label']
        return best_label


if __name__ == "__main__":
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)

    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    results_indep = evaluator.evaluate(
        model=DollarOne3DGestureRecognizer(),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        n_folds=10
    )
    
    results_dep = evaluator.evaluate(
        model=DollarOne3DGestureRecognizer(),
        df=df,
        evaluation_type="user-dependent",
        normalize=False,
        n_folds=10
    )

    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")