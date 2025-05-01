import numpy as np
from scipy.interpolate import interp1d
from hmmlearn import hmm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_domain

# -----------------------------------------------------------
# 1. Preprocessing
# -----------------------------------------------------------

def preprocess_sequence(seq, target_length=50):
    """Normalize and resample sequence"""
    # 1. Center to origin
    seq = seq - seq[0]

    # 2. Resample to fixed length
    original_length = seq.shape[0]
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)

    interpolator = interp1d(x_old, seq, axis=0, kind='linear')
    seq_resampled = interpolator(x_new)

    # 3. Scale to [-1, 1] range
    seq_normalized = 2 * (seq_resampled - np.min(seq_resampled)) / \
        (np.max(seq_resampled) - np.min(seq_resampled)) - 1

    return seq_normalized

# -----------------------------------------------------------
# 3. HMM Implementation
# -----------------------------------------------------------


class HMMClassifier:
    def __init__(self, n_states=5, n_iter=100):
        self.models = {}
        self.n_states = n_states
        self.n_iter = n_iter

    def preprocess(self, X):
        return [preprocess_sequence(seq) for seq in X]
    
    def fit(self, X, y):
        X= self.preprocess(X)
        for label in np.unique(y):
            model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter
            )
            sequences = [x for x, lbl in zip(X, y) if lbl == label]
            lengths = [len(s) for s in sequences]
            model.fit(np.vstack(sequences), lengths)
            self.models[label] = model

    def predict(self, X):
        X= self.preprocess(X)
        preds = []
        for seq in X:
            scores = []
            for label, model in self.models.items():
                try:
                    scores.append(model.score(seq))
                except:
                    scores.append(-np.inf)
            preds.append(list(self.models.keys())[np.argmax(scores)])
        return np.array(preds)


# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------

if __name__ == "__main__":
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    results_indep = evaluator.evaluate(
        model=HMMClassifier(),
        df=df,
        evaluation_type="user-independent",
        normalize=False,
        n_folds=10
    )
    
    results_dep = evaluator.evaluate(
        model=HMMClassifier(),
        df=df,
        evaluation_type="user-dependent",
        normalize=False,
        n_folds=10
    )

    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")