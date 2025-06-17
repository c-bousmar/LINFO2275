import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import extract_features_from_gesture, get_dataset_from_all_domains, get_dataset_from_domain


class LRGestureRecognizer:
    """
    Logistic Regression based gesture recognizer using statistical features.
    Uses a scikit-learn pipeline with standardization and L1 regularization.
    """
    
    def __init__(self):
        """
        Initialize the Logistic Regression Gesture Recognizer.
        Creates a pipeline with scaling and optimized logistic regression.
        """
        self.classifier = Pipeline([
            ('scaler', StandardScaler()),
            # Hyperparameters optimized with BayesSearchCV
            ('logistic', LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                C=2.5,
                penalty="l1"
            ))
        ])
    
    def extract_features_from_all_gestures(self, sequences):
        """
        Extract statistical features from all gesture sequences.
        
        @param sequences: List of gesture sequences
        @return: List of feature vectors
        """
        return [extract_features_from_gesture(sequence) for sequence in sequences]

    def fit(self, X_train, y_train):
        """
        Train the gesture recognizer on provided data.
        
        @param X_train: List of training gesture sequences
        @param y_train: List of training labels
        """
        X_train_features = self.extract_features_from_all_gestures(X_train)
        self.classifier.fit(np.array(X_train_features), np.array(y_train))

    def predict(self, X):
        """
        Predict gesture labels for new data.
        
        @param X: List of gesture sequences to classify
        @return: Predicted labels
        """
        X_features = self.extract_features_from_all_gestures(X)
        return self.classifier.predict(np.array(X_features))
    

if __name__ == '__main__':
    # Load dataset
    domain_id = 4
    df = get_dataset_from_all_domains("../Data/dataset.csv")

    # Initialize evaluator with verbose output
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate using user-independent cross-validation
    results_indep = evaluator.evaluate(
        model=LRGestureRecognizer(),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        n_folds=10
    )
    
    # Evaluate using user-dependent cross-validation
    results_dep = evaluator.evaluate(
        model=LRGestureRecognizer(),
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
    evaluator.save_results_to_csv(results_indep, f"../Results/LogisticRegression/_user_independent_all_domain.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/LogisticRegression/_user_dependent_all_domain.csv")