import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from GestureRecognizerEstimator import GestureRecognitionEvaluator

from datasets_utils import extract_features_from_gesture, get_dataset_from_domain


class TransformersGestureRecognizer:
    
    def __init__(self, epochs=100, batch_size=32, verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.fit_scaler = None
    
    def create_model(self, input_dim, num_classes, dropout_rate=0.4):
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
        
        # Output layer
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def extract_features_from_all_gestures(self, sequences):
        return [extract_features_from_gesture(sequence) for sequence in sequences]
    
    def fit(self, X_train, y_train):
        X_train_features = np.array(self.extract_features_from_all_gestures(X_train))
        self.fit_scaler = StandardScaler()
        X_train_features = self.fit_scaler.fit_transform(X_train_features)
        
        if isinstance(y_train[0], str):
            y_train = LabelEncoder().fit_transform(y_train)
        else:
            y_train = np.array(y_train)
        
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        input_dim = X_train_features.shape[1]
        
        self.model = self.create_model(
            input_dim=input_dim, 
            num_classes=num_classes,
            dropout_rate=0.4
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        self.model.fit(
            X_train_features, y_train, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0 if not self.verbose else 1
        )

    
    def predict(self, X_test):
        X_test_features = np.array(self.extract_features_from_all_gestures(X_test))
        X_test_features = self.fit_scaler.transform(X_test_features)
        y_pred_proba = self.model.predict(X_test_features, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred
    

if __name__ == "__main__":
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number=1)

    evaluator = GestureRecognitionEvaluator(verbose=True)
    
    results_indep = evaluator.evaluate(
        model=TransformersGestureRecognizer(epochs=100, batch_size=32, verbose=True),
        df=df,
        evaluation_type="user-independent",
        normalize=True,
        n_folds=10
    )
    
    results_dep = evaluator.evaluate(
        model=TransformersGestureRecognizer(epochs=100, batch_size=32, verbose=True),
        df=df,
        evaluation_type="user-dependent",
        normalize=True,
        n_folds=10
    )

    print(f"\nUser-Independent - Accuracy: {results_indep['mean_accuracy']:.2%} ± {results_indep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_indep['confusion_matrix']}")
    
    print(f"\nUser-Dependent - Accuracy: {results_dep['mean_accuracy']:.2%} ± {results_dep['std_accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results_dep['confusion_matrix']}")