import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from GestureRecognizerEstimator import GestureRecognitionEvaluator

from datasets_utils import extract_features_from_gesture, get_dataset_from_domain

class LRGestureRecognizer:
    
    def __init__(self):
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            # Found with BayesSearchCV
            ('logistic', LogisticRegression(solver="liblinear", max_iter=2000, C=2.5, penalty="l1"))
        ])
    
    def extract_features_from_all_gestures(self, sequences):
        return [extract_features_from_gesture(sequence) for sequence in sequences]

    def fit(self, X_train, y_train):
        X_train_features = self.extract_features_from_all_gestures(X_train)
        self.classifier.fit(np.array(X_train_features), np.array(y_train)
)

    def predict(self, X):
        X_features = self.extract_features_from_all_gestures(X)
        return self.classifier.predict(np.array(X_features))

if __name__ == '__main__':
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)

    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    results_indep = evaluator.evaluate(
        model=LRGestureRecognizer(),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    results_dep = evaluator.evaluate(
        model=LRGestureRecognizer(),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )

    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")