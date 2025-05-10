import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import get_dataset_from_all_domains, get_dataset_from_domain

class LSTMGestureRecognizer:
    """
    LSTM-based 3D gesture recognition system
    """
    
    def __init__(self, num_points=100, num_classes=20):
        """
        Initialize the LSTM Recognizer
        
        @param num_points: Number of resampled points per gesture
        @param num_classes: Number of gesture classes
        """
        self.num_points = num_points
        self.num_classes = num_classes
        self.le = LabelEncoder()
        self.model = self._build_model()
    
    def _build_model(self):
        """Construct LSTM network architecture"""
        model = Sequential([
            LSTM(128, input_shape=(self.num_points, 3), return_sequences=True),
            Dropout(0.5),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model
    
    def _resample(self, points):
        """Resample gesture to fixed-length sequence"""
        cumulative_dist = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        cumulative_dist = np.insert(cumulative_dist, 0, 0)
        
        if cumulative_dist[-1] == 0:
            return np.tile(points[0], (self.num_points, 1))
            
        return interp1d(cumulative_dist, points, axis=0)(
            np.linspace(0, cumulative_dist[-1], self.num_points)
        )

    def _preprocess(self, points):
        """Main preprocessing pipeline"""
        return self._resample(points)
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model"""
        # Preprocess all training data
        X_processed = np.array([self._preprocess(g) for g in X_train])
        
        # Encode labels
        y_encoded = self.le.fit_transform(y_train)
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        
        # Train model
        self.model.fit(
            X_processed, 
            y_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
    
    def predict(self, X):
        """Make predictions on new data"""
        X_processed = np.array([self._preprocess(g) for g in X])
        predictions = self.model.predict(X_processed, verbose=0)
        return self.le.inverse_transform(np.argmax(predictions, axis=1))
    
    def recognize(self, gesture):
        """Recognize single gesture"""
        processed = self._preprocess(gesture)
        prediction = self.model.predict(np.array([processed]), verbose=0)
        return self.le.inverse_transform([np.argmax(prediction)]).item()

if __name__ == "__main__":
    # Load dataset
    domain_id = 4
    df = get_dataset_from_all_domains("../Data/dataset.csv")
    
    # Initialize evaluator
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # User-independent evaluation
    results_indep = evaluator.evaluate(
        model=LSTMGestureRecognizer(),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10  # Reduced for faster training
    )
    
    # User-dependent evaluation
    results_dep = evaluator.evaluate(
        model=LSTMGestureRecognizer(),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )

    # Display results
    print("\nUser-Independent Results:")
    print(f"Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print("\nUser-Dependent Results:")
    print(f"Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")

    # Save results
    evaluator.save_results_to_csv(results_indep, f"../Results/LSTM/_user_independent_all_domain.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/LSTM/_user_dependent_all_domain.csv")
    