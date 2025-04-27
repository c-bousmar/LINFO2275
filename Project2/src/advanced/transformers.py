import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from datasets_utils import get_dataset_from_domain


def extract_features_from_gesture(sequence):
    """Extract statistical features from a gesture sequence."""
    # Basic Statistic Values
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)
    range_of_motion = max_vals - min_vals
    # skewness value greater than zero means that there is more weight in the right tail of the distribution
    skewness = skew(sequence, axis=0, bias=False)
    # kurtosis = fourth central moment divided by the square of the variance
    kurt = kurtosis(sequence, axis=0, bias=False)

    if len(sequence) > 1:
        # Velocity and their statistics
        velocity = np.diff(sequence, axis=0)
        mean_velocity = np.mean(velocity, axis=0)
        std_velocity = np.std(velocity, axis=0)
        max_velocity = np.max(np.abs(velocity), axis=0)
        energy_velocity = np.sum(velocity**2, axis=0) # Energy of the velocity
        sma_velocity = np.sum(np.abs(velocity)) / len(velocity) # Smoothness of the velocity

        # Path length (total distance traveled)
        path_length = np.sum(np.sqrt(np.sum(velocity**2, axis=1)))

        # Orientation change
        initial_vector = sequence[1] - sequence[0]
        final_vector = sequence[-1] - sequence[-2]
        cos_theta = np.dot(initial_vector, final_vector) / (np.linalg.norm(initial_vector) * np.linalg.norm(final_vector) + 1e-8)
        orientation_change = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    else:
        mean_velocity = np.zeros(3)
        std_velocity = np.zeros(3)
        max_velocity = np.zeros(3)
        energy_velocity = np.zeros(3)
        sma_velocity = 0
        path_length = 0
        orientation_change = 0

    if len(sequence) > 2:
        # Acceleration and their statistics
        acceleration = np.diff(velocity, axis=0)
        mean_acceleration = np.mean(acceleration, axis=0)
        std_acceleration = np.std(acceleration, axis=0)
        max_acceleration = np.max(np.abs(acceleration), axis=0)
        energy_acceleration = np.sum(acceleration**2, axis=0)
        sma_acceleration = np.sum(np.abs(acceleration)) / len(acceleration)
    else:
        mean_acceleration = np.zeros(3)
        std_acceleration = np.zeros(3)
        max_acceleration = np.zeros(3)
        energy_acceleration = np.zeros(3)
        sma_acceleration = 0

    # Concatenate all features into a single array
    features = np.concatenate([
        mean, std, min_vals, max_vals, range_of_motion,
        skewness, kurt,
        mean_velocity, std_velocity, max_velocity, energy_velocity, [sma_velocity],
        mean_acceleration, std_acceleration, max_acceleration, energy_acceleration, [sma_acceleration],
        [path_length, orientation_change]
    ])

    return features


def create_transformers_model(input_dim, num_classes, dropout_rate=0.4):
    """Create a model that uses only statistical features."""
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


def prepare_gesture_features(df):
    """Extract statistical features from gestures."""
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    statistical_features = []
    labels = []
    subject_ids = []
    trial_ids = []
    
    for (subject_id, target, trial_id), group in gesture_groups:
        # Sort by time and get coordinates
        sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
        
        # Z-score normalize each sequence
        scaler = StandardScaler()
        normalized_sequence = scaler.fit_transform(sequence)
        
        # Extract statistical features
        stats_feature_vector = extract_features_from_gesture(normalized_sequence)
        statistical_features.append(stats_feature_vector)
        
        labels.append(target)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)
    
    # Standardize statistical features
    statistical_features = np.array(statistical_features)
    stats_scaler = StandardScaler()
    statistical_features = stats_scaler.fit_transform(statistical_features)
    
    # Encode labels if they are strings
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    else:
        labels = np.array(labels)
    
    return statistical_features, labels, np.array(subject_ids), np.array(trial_ids)


def user_independent_transformers_score(domain_number=1, epochs=100, batch_size=32, verbose=True):
    """Evaluate features-only model with user-independent cross-validation."""
    dataset_path = f"../Data/dataset.csv"
    if verbose:
        print(f"Loading dataset from {dataset_path}")
    
    # Load the dataset
    df = get_dataset_from_domain(dataset_path, domain_number)
    
    # Get unique subject IDs
    subject_ids = sorted(df['subject_id'].unique())
    
    # Initialize results storage
    user_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting leave-one-user-out cross-validation for domain {domain_number}")
        print(f"Total users: {len(subject_ids)}")
    
    # Iterate through each user
    for test_subject in tqdm(subject_ids, desc="Users", disable=not verbose):
        # Split data into training and test sets
        train_df = df[df['subject_id'] != test_subject]
        test_df = df[df['subject_id'] == test_subject]
        
        # Prepare the training and test sets with statistical features only
        X_train, y_train, _, _ = prepare_gesture_features(train_df)
        X_test, y_test, _, _ = prepare_gesture_features(test_df)
        
        # Get unique classes
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        
        # Create and train the features-only model
        input_dim = X_train.shape[1]  # Number of statistical features
        
        current_model = create_transformers_model(
            input_dim=input_dim, 
            num_classes=num_classes,
            dropout_rate=0.3
        )
        
        # Use early stopping and learning rate reduction
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train the model with only statistical features
        current_model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0 if not verbose else 1
        )

        # Make predictions
        y_pred_proba = current_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        user_accuracies.append(accuracy)
        
        # Store true and predicted labels for confusion matrix
        all_true_labels.extend(y_test)
        all_pred_labels.extend(y_pred)
        
        if verbose:
            print(f"User {test_subject} accuracy: {accuracy:.4f}")
        
        # Clean up to free memory
        tf.keras.backend.clear_session()
    
    # Calculate overall metrics
    mean_accuracy = np.mean(user_accuracies)
    std_accuracy = np.std(user_accuracies)
    
    # Generate confusion matrix
    unique_targets = np.unique(all_true_labels)
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_targets)
    report = classification_report(all_true_labels, all_pred_labels)
    
    if verbose:
        print(f"\nUser-independent evaluation results for domain {domain_number}:")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Features-Only Model Domain {domain_number} (User-Independent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("advanced/results/features_only_confusion_matrix.pdf", format="pdf")
        plt.close()
    
    # Return results
    return {
        'user_accuracies': user_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }


def user_dependent_transformers_score(domain_number=1, epochs=100, batch_size=32, verbose=True):
    """Evaluate features-only model with trial-based cross-validation."""
    # Load the dataset
    dataset_path = f"../Data/dataset.csv"
    if verbose:
        print(f"Loading dataset from {dataset_path}")
    
    df = get_dataset_from_domain(dataset_path, domain_number)
    
    # Prepare a cross-validation based on trial ids
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    trial_ids = []
    
    for (_, _, trial_id), _ in gesture_groups:
        if trial_id not in trial_ids:
            trial_ids.append(trial_id)
    
    trial_ids = sorted(trial_ids)
    
    # Initialize results storage
    fold_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting trial-based cross-validation for domain {domain_number}")
        print(f"Total trials: {len(trial_ids)}")
    
    # Use trial IDs for cross-validation
    for test_trial_id in tqdm(trial_ids, desc="Trials", disable=not verbose):
        # Split data into training and test sets
        train_df = df[df['trial_id'] != test_trial_id]
        test_df = df[df['trial_id'] == test_trial_id]
        
        # Prepare the training and test sets with statistical features only
        X_train, y_train, _, _ = prepare_gesture_features(train_df)
        X_test, y_test, _, _ = prepare_gesture_features(test_df)
        
        # Get unique classes
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        
        # Create and train the features-only model
        input_dim = X_train.shape[1]  # Number of statistical features
        
        current_model = create_transformers_model(
            input_dim=input_dim, 
            num_classes=num_classes,
            dropout_rate=0.3
        )
        
        # Use early stopping and learning rate reduction
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
        ]
        
        # Train the model with only statistical features
        current_model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0 if not verbose else 1
        )

        # Make predictions
        y_pred_proba = current_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        # Store true and predicted labels for confusion matrix
        all_true_labels.extend(y_test)
        all_pred_labels.extend(y_pred)
        
        if verbose:
            print(f"Trial {test_trial_id} accuracy: {accuracy:.4f}")
        
        # Clean up to free memory
        tf.keras.backend.clear_session()
    
    # Calculate overall metrics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    # Generate confusion matrix
    unique_targets = np.unique(all_true_labels)
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_targets)
    report = classification_report(all_true_labels, all_pred_labels)
    
    if verbose:
        print(f"\nUser-dependent evaluation results for domain {domain_number}:")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Features-Only Model Domain {domain_number} (User-Dependent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("advanced/results/features_only_user_dependent_confusion_matrix.pdf", format="pdf")
        plt.close()
    
    # Return results
    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }


if __name__ == "__main__":
    
    domain_number = 1
    epochs = 100
    batch_size = 32
    verbose = True
    
    print("Running user-independent evaluation with Features-Only model...")
    _, mean_acc_indep, mean_std_indep, _ = user_independent_transformers_score(domain_number=domain_number, epochs=epochs, batch_size=batch_size, verbose=verbose)[1]
    
    print("\nRunning user-dependent evaluation with Features-Only model...")
    _, mean_acc_dep, std_acc_dep, _ = user_dependent_transformers_score(domain_number=domain_number, epochs=epochs, batch_size=batch_size, verbose=verbose)[1]
    
    print(f"\nUser-independent mean accuracy: {mean_acc_indep:.4f} ± {mean_std_indep:.4f}")
    print(f"User-dependent mean accuracy: {mean_acc_dep:.4f} ± {std_acc_dep:.4f}")