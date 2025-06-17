import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from KNN_Classifier import KNN_Classifier
from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain
from distance_metrics import euclidean_distance, dtw_distance, edit_distance, lcs_distance

class KNNWithCustomDist:
    """
    Wrapper for KNN Classifier that simplifies using different distance metrics
    for gesture recognition tasks.
    """
    
    def __init__(self, k=1, distance_function=euclidean_distance):
        """
        Initialize the KNN with custom distance function.
        
        @param k: Number of nearest neighbors to consider
        @param distance_function: Distance function to use for comparing gestures
        """
        self.k = k
        self.distance_function = distance_function
        self.classifier = KNN_Classifier(k=k, distance_function=distance_function)
    
    def fit(self, X_train, y_train):
        """
        Train the classifier on provided data.
        
        @param X_train: List of training gesture sequences
        @param y_train: List of training labels
        """
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Predict labels for test data.
        
        @param X_test: List of test gesture sequences
        @return: List of predicted labels
        """
        return self.classifier.predict(X_test)
    
if __name__ == '__main__':
    
    k = 1
    domain_id = 4
    distance_function = euclidean_distance
    use_string_repr = False
    
    # Get dataset
    if use_string_repr:
        df = get_dataset_from_domain(f"../Data/labelled_gestures_{domain_id}.csv", domain_number=domain_id)
    else:
        df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_id)
    
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
    
    # Display results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")
    
    # Save results
    evaluator.save_results_to_csv(results_indep, f"../Results/KNN/_user_independent_domain{domain_id}_k{k}_dist_{distance_function.__name__}.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/KNN/_user_dependent_domain{domain_id}_k{k}_dist_{distance_function.__name__}.csv")