import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GestureRecognizerEstimator import GestureRecognitionEvaluator
from datasets_utils import extract_features_from_gesture, get_dataset_from_domain

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class DenseNetGestureRecognizer:
    """
    Neural network-based gesture recognizer using statistical features.
    Implements a multi-layer neural network for gesture classification.
    """
    
    def __init__(self, epochs=100, batch_size=32, verbose=True):
        """
        Initialize the DenseNet Gesture Recognizer.
        
        @param epochs: Number of training epochs
        @param batch_size: Batch size for training
        @param verbose: Whether to display training progress
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.fit_scaler = None
        self.label_encoder = None
        self.is_string_labels = False
    
    def create_model(self, input_dim, num_classes, dropout_rate=0.4):
        """
        Create the neural network model architecture.
        
        @param input_dim: Dimension of input features
        @param num_classes: Number of gesture classes
        @param dropout_rate: Dropout rate for regularization
        @return: Compiled Keras model
        """
        # Input for statistical features
        inputs = layers.Input(shape=(input_dim,), name="stat_input")
        
        # Hidden layers
        x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer with softmax activation for classification
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with Adam optimizer
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
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
        # Extract features from all gestures
        X_train_features = np.array(self.extract_features_from_all_gestures(X_train))
        
        # Standardize features
        self.fit_scaler = StandardScaler()
        X_train_features = self.fit_scaler.fit_transform(X_train_features)
        
        # Encode labels if they are strings (for domain 4)  
        if isinstance(y_train[0], str):
            self.is_string_labels = True
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
        else:
            self.is_string_labels = False
            y_train_encoded = np.array(y_train)
        
        # Determine model dimensions
        unique_classes = np.unique(y_train_encoded)
        num_classes = len(unique_classes)
        input_dim = X_train_features.shape[1]
        
        # Create the neural network model
        self.model = self.create_model(
            input_dim=input_dim, 
            num_classes=num_classes,
            dropout_rate=0.4
        )
        
        # Set up callbacks for training optimization
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train the model
        self.model.fit(
            X_train_features, y_train_encoded, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0 if not self.verbose else 1
        )

    def predict(self, X_test):
        """
        Predict gesture labels for new data.
        
        @param X_test: List of gesture sequences to classify
        @return: Predicted class indices
        """
        # Extract and standardize features
        X_test_features = np.array(self.extract_features_from_all_gestures(X_test))
        X_test_features = self.fit_scaler.transform(X_test_features)
        
       # Generate predictions (numeric indices)
        y_pred_proba = self.model.predict(X_test_features, verbose=0)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        
        # Convert back to original label format if necessary
        if self.is_string_labels and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_pred_indices)
        else:
            return y_pred_indices
    

if __name__ == "__main__":
    # Load dataset
    domain_id = 4
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=domain_id)

    # Initialize evaluator with verbose output
    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    # Evaluate using user-independent cross-validation
    results_indep = evaluator.evaluate(
        model=DenseNetGestureRecognizer(epochs=100, batch_size=32, verbose=True),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    # Evaluate using user-dependent cross-validation
    results_dep = evaluator.evaluate(
        model=DenseNetGestureRecognizer(epochs=100, batch_size=32, verbose=True),
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
    evaluator.save_results_to_csv(results_indep, f"../Results/DenseNet/_user_independent_domain{domain_id}.csv")
    evaluator.save_results_to_csv(results_dep, f"../Results/DenseNet/_user_dependent_domain{domain_id}.csv")