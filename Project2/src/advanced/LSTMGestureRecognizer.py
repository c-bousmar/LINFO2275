import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder

from datasets_utils import get_dataset_from_domain

from GestureRecognizerEstimator import GestureRecognitionEvaluator


class LSTMGestureRecognizer:
    
    def __init__(self, epochs=50, batch_size=32, verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.max_length = None
        self.label_encoder = None
    
    def create_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
        
        return model
    
    def find_best_max_length(self, sequences, percentile=95):
        """
        Find the best max length for padding sequences
        """
        lengths = [len(seq) for seq in sequences]
        
        if self.verbose:
            print(f"Min length: {np.min(lengths)}")
            print(f"Max length: {np.max(lengths)}")
            print(f"Mean length: {np.mean(lengths):.2f}")
            print(f"Median length: {np.median(lengths)}")
        
        best_max_length = int(np.percentile(lengths, percentile))
        if self.verbose:
            print(f"Using max_length at {percentile}% percentile: {best_max_length}")
        
        return best_max_length
    
    def pad_sequences(self, sequences, max_length):
        """
        Pad or truncate sequences to a fixed length
        """
        padded_sequences = []
        
        for sequence in sequences:
            if len(sequence) > max_length:
                # Truncate
                padded_sequence = sequence[:max_length]
            else:
                # Pad
                padding_length = max_length - len(sequence)
                if padding_length > 0:
                    padding = np.zeros((padding_length, sequence.shape[1]))
                    padded_sequence = np.vstack([sequence, padding])
                else:
                    padded_sequence = sequence
            
            padded_sequences.append(padded_sequence)
        
        return np.array(padded_sequences)
    
    def fit(self, X_train, y_train):
        """
        Fit the model to training data
        """
        # Find the best max length for padding
        self.max_length = self.find_best_max_length(X_train)
        
        # Pad sequences to fixed length
        X_train_padded = self.pad_sequences(X_train, self.max_length)
        
        # Encode labels if they are strings
        if isinstance(y_train[0], str):
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_train_encoded = np.array(y_train)
        
        # Get number of classes
        unique_classes = np.unique(y_train_encoded)
        num_classes = len(unique_classes)
        
        # Create model
        input_shape = (X_train_padded.shape[1], X_train_padded.shape[2])
        self.model = self.create_model(input_shape, num_classes)
        
        # Use early stopping to prevent overfitting
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        # Train model
        self.model.fit(
            X_train_padded, y_train_encoded, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0 if not self.verbose else 1
        )
    
    def predict(self, X_test):
        """
        Predict labels for test data
        """
        # Pad sequences to same length as training data
        X_test_padded = self.pad_sequences(X_test, self.max_length)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test_padded, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # If we used a label encoder, transform predictions back
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred


if __name__ == "__main__":
    # Get dataset
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)
    
    # Create evaluator
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate with user-independent protocol
    results_indep = evaluator.evaluate(
        model=LSTMGestureRecognizer(epochs=50, batch_size=32, verbose=True),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    # Evaluate with user-dependent protocol
    results_dep = evaluator.evaluate(
        model=LSTMGestureRecognizer(epochs=50, batch_size=32, verbose=True),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )
    
    # Print results
    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrix(
        results_indep, 
        title="Confusion Matrix - LSTM (User-Independent)",
        filename="advanced/results/lstm_user_independent_confusion_matrix.pdf"
    )
    
    evaluator.plot_confusion_matrix(
        results_dep, 
        title="Confusion Matrix - LSTM (User-Dependent)",
        filename="advanced/results/lstm_user_dependent_confusion_matrix.pdf"
    )