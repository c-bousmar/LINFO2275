import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain

from scipy.interpolate import interp1d
from hmmlearn import hmm
import numpy as np

class HMMClassifier:
    """
    Hidden Markov Model Classifier for gesture recognition.
    Uses separate Gaussian HMM models for each gesture class.
    """
    
    def __init__(self, n_states=5, n_iter=100):
        """
        Initialize the HMM Classifier.
        
        @param n_states: Number of hidden states in each HMM
        @param n_iter: Maximum number of iterations for training
        """
        self.models = {}  # Dictionary to store one HMM per gesture class
        self.n_states = n_states
        self.n_iter = n_iter
        
    def preprocess_sequence(self, seq, target_length=50):
        """
        Normalize and resample a single sequence.
        
        @param seq: Input sequence (gesture)
        @param target_length: Target length after resampling
        @return: Normalized and resampled sequence
        """
        # 1. Center to origin by subtracting first point
        seq = seq - seq[0]

        # 2. Resample to fixed length using linear interpolation
        original_length = seq.shape[0]
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)

        interpolator = interp1d(x_old, seq, axis=0, kind='linear')
        seq_resampled = interpolator(x_new)

        # 3. Scale to [-1, 1] range for better numerical stability
        seq_normalized = 2 * (seq_resampled - np.min(seq_resampled)) / \
            (np.max(seq_resampled) - np.min(seq_resampled)) - 1

        return seq_normalized

    def preprocess(self, X):
        """
        Preprocess all sequences in a dataset.
        
        @param X: List of sequences to preprocess
        @return: List of preprocessed sequences
        """
        return [self.preprocess_sequence(seq) for seq in X]
    
    def fit(self, X, y):
        """
        Train HMM models for each gesture class.
        
        @param X: List of training gesture sequences
        @param y: List of training labels
        """
        # Preprocess all sequences
        X = self.preprocess(X)
        
        # Train one HMM per gesture class
        for label in np.unique(y):
            # Create a new HMM model
            model = hmm.GaussianHMM(
                n_components=self.n_states,  # Number of hidden states
                covariance_type="diag",      # Diagonal covariance for efficiency
                n_iter=self.n_iter           # Maximum number of EM iterations
            )
            
            # Extract sequences for current class
            sequences = [x for x, lbl in zip(X, y) if lbl == label]
            lengths = [len(s) for s in sequences]
            
            # Fit the model on stacked sequences
            model.fit(np.vstack(sequences), lengths)
            
            # Store the model
            self.models[label] = model

    def predict(self, X):
        """
        Predict gesture labels for new data.
        
        @param X: List of gesture sequences to classify
        @return: Predicted labels
        """
        # Preprocess test sequences
        X = self.preprocess(X)
        
        preds = []
        for seq in X:
            scores = []
            # Calculate log-likelihood score for each class model
            for label, model in self.models.items():
                try:
                    scores.append(model.score(seq))
                except:
                    # Handle potential numerical issues
                    scores.append(-np.inf)
            
            # Predict the class with the highest score
            preds.append(list(self.models.keys())[np.argmax(scores)])
            
        return np.array(preds)
    

if __name__ == '__main__':
    # Load dataset
    domain_id = 1
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_id)

    # Initialize evaluator with verbose output
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate using user-independent cross-validation
    results_indep = evaluator.evaluate(
        model=HMMClassifier(),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        n_folds=10
    )
    
    # Evaluate using user-dependent cross-validation
    results_dep = evaluator.evaluate(
        model=HMMClassifier(),
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
    evaluator.save_results_to_csv(results_indep, f"../Results/HMM/_user_independent_domain{domain_id}.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/HMM/_user_dependent_domain{domain_id}.csv")