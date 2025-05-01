import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from advanced.LRGestureRecognizer import LRGestureRecognizer
from advanced.DollarOne3DGestureRecognizer import DollarOne3DGestureRecognizer
from advanced.LSTMGestureRecognizer import LSTMGestureRecognizer
from advanced.TransformersGestureRecognizer import TransformersGestureRecognizer
from advanced.FastMDSWithClassifier import FastMDSWithClassifier

estimator = GestureRecognitionEvaluator()

df = estimator.get_dataset_from_domain("../Data/dataset.csv", domain_number=1)
models = {
    "$1 Recognizer" : DollarOne3DGestureRecognizer(),
    "LR" : LRGestureRecognizer(),
    "LSTM" : LSTMGestureRecognizer(),
    "Transformers" : TransformersGestureRecognizer(),
    "FastMDS" : FastMDSWithClassifier()
}

estimator.compare_models(models, df, evaluation_types=['user-independent', 'user-dependent'], n_folds=10)