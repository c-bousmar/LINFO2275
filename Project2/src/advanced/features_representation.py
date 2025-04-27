import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import skew, kurtosis

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from datasets_utils import get_dataset_from_domain


def prepare_gesture_sequences(df, normalize=True):
    """Prepare gesture sequences from the dataframe."""
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    sequences = []
    labels = []
    subject_ids = []
    trial_ids = []
    
    for (subject_id, target, trial_id), group in gesture_groups:
        # Sort by time and get coordinates
        sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
        
        # Normalize if requested
        if normalize:
            # Z-score normalization for HMM
            scaler = StandardScaler()
            sequence = scaler.fit_transform(sequence)
        
        sequences.append(sequence)
        labels.append(target)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)
    
    return sequences, labels, subject_ids, trial_ids


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


def extract_features_from_all_gestures(sequences):
    """Extract features from a list of gesture sequences."""
    return [extract_features_from_gesture(sequence) for sequence in sequences]


class HMMGestureClassifier:
    """HMM-based classifier for gesture recognition."""
    
    def __init__(self, n_components=5, covariance_type='diag', n_iter=2000):
        """Initialize the HMM classifier.
        
        Args:
            n_components: Number of hidden states in the HMM
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied', 'spherical')
            n_iter: Number of iterations for HMM training
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}
        self.classes = []
    
    def fit(self, X, y):
        """Fit an HMM for each class in the training data.
        
        Args:
            X: List of sequences (each a numpy array of shape [time_steps, features])
            y: List of labels for each sequence
        """
        self.classes = np.unique(y)
        
        # Extract statistical features from each sequence
        X_features = extract_features_from_all_gestures(X)
        
        # Prepare training data for each class
        for cls in self.classes:
            # Get all features for this class
            class_features = [X_features[i] for i in range(len(X_features)) if y[i] == cls]
            
            # Reshape features for HMM (adding a time dimension of length 1)
            # This is needed because HMM expects sequences, but we now have fixed-length feature vectors
            X_combined = np.array([feature.reshape(1, -1) for feature in class_features])
            lengths = [1] * len(class_features)  # Each "sequence" is just one feature vector
            X_combined = np.vstack(X_combined)  # Stack all the feature vectors
            
            # Initialize and train HMM for this class
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42
            )
            
            # Train the model
            try:
                model.fit(X_combined, lengths)
                self.models[cls] = model
            except Exception as e:
                print(f"Error training HMM for class {cls}: {e}")
                # Create a fallback model in case of issues
                self.models[cls] = hmm.GaussianHMM(
                    n_components=2,
                    covariance_type='spherical',
                    n_iter=self.n_iter,
                    random_state=42
                )
                self.models[cls].fit(X_combined, lengths)
    
    def predict(self, X):
        """Predict class labels for each sequence in X.
        
        Args:
            X: List of sequences (each a numpy array of shape [time_steps, features])
            
        Returns:
            Array of predicted class labels
        """
        if not self.models:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Extract statistical features
        X_features = extract_features_from_all_gestures(X)
        
        predictions = []
        
        for feature in X_features:
            # Reshape feature for HMM (adding a time dimension of length 1)
            feature_sequence = feature.reshape(1, -1)
            
            # Compute log likelihood for each class model
            log_likelihoods = {}
            for cls, model in self.models.items():
                try:
                    log_likelihoods[cls] = model.score(feature_sequence)
                except Exception as e:
                    print(f"Error scoring sequence with class {cls} model: {e}")
                    log_likelihoods[cls] = float('-inf')
            
            # Predict the class with highest log likelihood
            if log_likelihoods:
                best_class = max(log_likelihoods, key=log_likelihoods.get)
                predictions.append(best_class)
            else:
                # Fallback to most common class if scoring failed for all models
                predictions.append(self.classes[0])
        
        return np.array(predictions)


def optimize_hmm_params(X_train, y_train, X_val, y_val):
    """Find the best HMM parameters using validation data."""
    best_accuracy = 0
    best_params = {}
    
    # Parameter grid to search
    param_grid = {
        'n_components': [3, 5, 7],
        'covariance_type': ['diag', 'full', 'tied']
    }
    
    # Simple grid search
    for n_components in param_grid['n_components']:
        for covariance_type in param_grid['covariance_type']:
            print(f"Testing HMM with {n_components} states, {covariance_type} covariance...")
            
            # Initialize and train model
            model = HMMGestureClassifier(
                n_components=n_components,
                covariance_type=covariance_type
            )
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"Validation accuracy: {accuracy:.4f}")
            
            # Update best parameters if improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'n_components': n_components,
                    'covariance_type': covariance_type
                }
    
    print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy:.4f}")
    return best_params


def user_independent_evaluation(domain_number=1, optimize_params=False, verbose=True):
    """Evaluate HMM with leave-one-user-out cross-validation."""
    # Load the dataset
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number)
    
    # Prepare sequences
    sequences, labels, subject_ids, _ = prepare_gesture_sequences(df)
    
    # Get unique subjects and targets
    unique_subjects = sorted(set(subject_ids))
    unique_targets = sorted(set(labels))
    
    if verbose:
        print(f"Total gestures: {len(sequences)}")
        print(f"Total users: {len(unique_subjects)}")
        print(f"Total classes: {len(unique_targets)}")
    
    # Initialize results
    user_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    # Default HMM parameters
    hmm_params = {
        'n_components': 5,
        'covariance_type': 'diag',
        'n_iter': 2000
    }
    
    # Create a mapping from sequences to data
    sequence_data = list(zip(sequences, labels, subject_ids))
    
    # Leave-one-user-out cross-validation
    for test_subject in tqdm(unique_subjects, desc="User cross-validation", disable=not verbose):
        # Split data
        train_data = [(seq, label) for seq, label, subj in sequence_data if subj != test_subject]
        test_data = [(seq, label) for seq, label, subj in sequence_data if subj == test_subject]
        
        X_train = [item[0] for item in train_data]
        y_train = [item[1] for item in train_data]
        X_test = [item[0] for item in test_data]
        y_test = [item[1] for item in test_data]
        
        # If requested, optimize HMM parameters using a validation set
        if optimize_params:
            # Split training data for validation
            val_size = int(len(X_train) * 0.2)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            # Find best parameters
            best_params = optimize_hmm_params(X_train, y_train, X_val, y_val)
            hmm_params.update(best_params)
        
        # Train HMM classifier
        classifier = HMMGestureClassifier(**hmm_params)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        user_accuracies.append(accuracy)
        
        # Store for confusion matrix
        all_true_labels.extend(y_test)
        all_pred_labels.extend(y_pred)
        
        if verbose:
            print(f"User {test_subject} accuracy: {accuracy:.4f}")
    
    # Calculate overall metrics
    mean_accuracy = np.mean(user_accuracies)
    std_accuracy = np.std(user_accuracies)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_targets)
    
    if verbose:
        print(f"\nUser-independent evaluation results (HMM with Statistical Features):")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_pred_labels, labels=unique_targets))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (HMM with Statistical Features, User-Independent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'advanced/results/hmm_stat_features_user_independent_domain{domain_number}.pdf', format="pdf")
        plt.close()
    
    # Return results
    return {
        'user_accuracies': user_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix,
        'params': hmm_params
    }


def user_dependent_evaluation(domain_number=1, optimize_params=False, verbose=True):
    """Evaluate HMM with trial-based cross-validation (user-dependent)."""
    # Load the dataset
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number)
    
    # Prepare sequences
    sequences, labels, subject_ids, trial_ids = prepare_gesture_sequences(df)
    
    # Get unique trials and targets
    unique_trials = sorted(set(trial_ids))
    unique_targets = sorted(set(labels))
    
    if verbose:
        print(f"Total gestures: {len(sequences)}")
        print(f"Total trials: {len(unique_trials)}")
        print(f"Total classes: {len(unique_targets)}")
    
    # Initialize results
    trial_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    # Default HMM parameters
    hmm_params = {
        'n_components': 5,
        'covariance_type': 'diag',
        'n_iter': 2000
    }
    
    # Create a mapping from sequences to data
    sequence_data = list(zip(sequences, labels, subject_ids, trial_ids))
    
    # Leave-one-trial-out cross-validation
    for test_trial in tqdm(unique_trials, desc="Trial cross-validation", disable=not verbose):
        # Split data
        train_data = [(seq, label) for seq, label, _, trial in sequence_data if trial != test_trial]
        test_data = [(seq, label) for seq, label, _, trial in sequence_data if trial == test_trial]
        
        X_train = [item[0] for item in train_data]
        y_train = [item[1] for item in train_data]
        X_test = [item[0] for item in test_data]
        y_test = [item[1] for item in test_data]
        
        # If requested, optimize HMM parameters using a validation set
        if optimize_params and len(X_train) > 10:  # Only optimize if we have enough data
            # Split training data for validation
            val_size = max(1, int(len(X_train) * 0.2))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]
            
            # Find best parameters
            best_params = optimize_hmm_params(X_train, y_train, X_val, y_val)
            hmm_params.update(best_params)
        
        # Train HMM classifier
        classifier = HMMGestureClassifier(**hmm_params)
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        trial_accuracies.append(accuracy)
        
        # Store for confusion matrix
        all_true_labels.extend(y_test)
        all_pred_labels.extend(y_pred)
        
        if verbose:
            print(f"Trial {test_trial} accuracy: {accuracy:.4f}")
    
    # Calculate overall metrics
    mean_accuracy = np.mean(trial_accuracies)
    std_accuracy = np.std(trial_accuracies)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_targets)
    
    if verbose:
        print(f"\nUser-dependent evaluation results (HMM with Statistical Features):")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_pred_labels, labels=unique_targets))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (HMM with Statistical Features, User-Dependent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'advanced/results/hmm_stat_features_user_dependent_domain{domain_number}.pdf', format="pdf")
        plt.close()
    
    # Return results
    return {
        'trial_accuracies': trial_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix,
        'params': hmm_params
    }

if __name__ == "__main__":
    print("Running user-independent evaluation with HMM and Statistical Features...")
    ui_results = user_independent_evaluation(domain_number=1, optimize_params=False)

    print("\nRunning user-dependent evaluation with HMM and Statistical Features...")
    ud_results = user_dependent_evaluation(domain_number=1, optimize_params=False)

    print("\nComparison of results:")
    print(f"User-independent accuracy: {ui_results['mean_accuracy']:.4f} ± {ui_results['std_accuracy']:.4f}")
    print(f"User-dependent accuracy: {ud_results['mean_accuracy']:.4f} ± {ud_results['std_accuracy']:.4f}")