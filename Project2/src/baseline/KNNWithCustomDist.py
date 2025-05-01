import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from KNN_Classifier import KNN_Classifier
from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain
from distance_metrics import euclidean_distance, dtw_distance, edit_distance, lcs_distance

class KNNWithCustomDist:
    
    def __init__(self, k=1, distance_function=euclidean_distance):
        self.k = k
        self.distance_function = distance_function
        self.classifier = KNN_Classifier(k=k, distance_function=distance_function)
    
    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.classifier.predict(X_test)
    
if __name__ == '__main__':
    
    use_string_repr = True
    k = 1
    domain_number = 1
    distance_function = edit_distance
    
    # Get dataset
    if use_string_repr:
        df = get_dataset_from_domain(f"../Data/labelled_gestures_{domain_number}.csv", domain_number=domain_number)
    else:
        df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_number)
    
    # Create evaluator
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate with user-independent protocol
    results_indep = evaluator.evaluate(
        model=KNNWithCustomDist(k=k, distance_function=distance_function),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        use_string_repr=use_string_repr,
        n_folds=10
    )
    
    # Evaluate with user-dependent protocol
    results_dep = evaluator.evaluate(
        model=KNNWithCustomDist(k=k, distance_function=distance_function),
        df=df,
        evaluation_type="user-dependent",
        normalize=False,
        use_string_repr=use_string_repr,
        n_folds=10
    )
    
    # Print results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")