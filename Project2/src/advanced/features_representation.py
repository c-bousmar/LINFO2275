import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.stats import skew, kurtosis

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from datasets_utils import get_dataset_from_domain

def prepare_gesture_sequences(df, normalize=True):
    # Group the dataset by several features : subject_id, target and trial_id
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    sequences, labels, subject_ids, trial_ids= [], [], [], []
    
    for (subject_id, target, trial_id), group in gesture_groups:
        # Sort the sequences by time and get the coordinates only
        sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
        
        # Normalize if requested
        if normalize:
            min_vals = np.min(sequence, axis=0)
            max_vals = np.max(sequence, axis=0)
            range_vals = np.maximum(max_vals - min_vals, 1e-10)
            sequence = (sequence - min_vals) / range_vals
        
        sequences.append(sequence)
        labels.append(target)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)
    
    return sequences, labels, subject_ids, trial_ids


def extract_features_from_gesture(sequence):
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
    return [extract_features_from_gesture(sequence) for sequence in sequences]


def user_independent_evaluation_with_representations(classifier, domain_number=1, verbose=True):
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
    
    # Load the dataset
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number)
    
    # Prepare sequences
    sequences, labels, subject_ids, _ = prepare_gesture_sequences(df, normalize=True)
    
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
    
    if verbose:
        print("Extracting features directly from gesture sequences...")
    
    # Extract features from all sequences
    representations = extract_features_from_all_gestures(sequences)
    
    # Create a mapping from sequences to data
    sequence_data = list(zip(representations, labels, subject_ids))
    
    # Leave-one-user-out cross-validation
    for test_subject in tqdm(unique_subjects, desc="User cross-validation", disable=not verbose):
        train_data = [(rep, label) for rep, label, subj in sequence_data if subj != test_subject]
        test_data = [(rep, label) for rep, label, subj in sequence_data if subj == test_subject]
        
        X_train = np.array([rep for rep, _ in train_data])
        y_train = np.array([label for _, label in train_data])
        X_test = np.array([rep for rep, _ in test_data])
        y_test = np.array([label for _, label in test_data])
        
        # Train classifier
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
        print(f"\nUser-independent evaluation results:")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_pred_labels, labels=unique_targets))
        
        classifier_type = list(classifier.named_steps.keys())[1]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (Features, {classifier_type})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'advanced/results/features_{classifier_type}_domain{domain_number}.pdf', format="pdf")
        plt.close()
    
    # Return results
    return {
        'user_accuracies': user_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix,
        'representations': representations,
        'labels': labels,
        'subject_ids': subject_ids
    }
    
def user_dependent_evaluation_with_representations(classifier, domain_number=1, verbose=True):
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
 
    # Load the dataset
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number)
    
    # Prepare sequences
    sequences, labels, subject_ids, trial_ids = prepare_gesture_sequences(df)
    
    # Get unique subjects, targets and trials
    unique_subjects = sorted(set(subject_ids))
    unique_targets = sorted(set(labels))
    unique_trials = sorted(set(trial_ids))
    
    if verbose:
        print(f"Total gestures: {len(sequences)}")
        print(f"Total users: {len(unique_subjects)}")
        print(f"Total classes: {len(unique_targets)}")
        print(f"Total trials: {len(unique_trials)}")
    
    # Initialize results
    trial_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print("Extracting features from gesture sequences...")
    
    # Extract features from all sequences
    representations = extract_features_from_all_gestures(sequences)
    
    # Create a mapping from sequences to data
    sequence_data = list(zip(representations, labels, subject_ids, trial_ids))
    
    # Leave-one-trial-out cross-validation
    for test_trial in tqdm(unique_trials, desc="Trial cross-validation", disable=not verbose):
        # Split data by trial
        train_data = [(rep, label, subj) for rep, label, subj, trial in sequence_data if trial != test_trial]
        test_data = [(rep, label, subj) for rep, label, subj, trial in sequence_data if trial == test_trial]
        
        X_train = np.array([rep for rep, _, _ in train_data])
        y_train = np.array([label for _, label, _ in train_data])
        X_test = np.array([rep for rep, _, _ in test_data])
        y_test = np.array([label for _, label, _ in test_data])
        
        # Train classifier
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
        print(f"\nUser-dependent evaluation results:")
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_true_labels, all_pred_labels, labels=unique_targets))
        
        classifier_type = list(classifier.named_steps.keys())[1]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (Features, {classifier_type}, User-Dependent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'advanced/results/features_user_dependent_{classifier_type}_domain{domain_number}.pdf', format="pdf")
        plt.close()
    
    # Return results
    return {
        'trial_accuracies': trial_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix,
        'representations': representations,
        'labels': labels,
        'subject_ids': subject_ids
    }

if __name__ == '__main__':
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        # Found with BayesSearchCV
        ('logistic', LogisticRegression(solver="liblinear", max_iter=2000, C=2.5, penalty="l1"))
    ])
    
    # Run both evaluations
    print("Running user-independent evaluation...")
    user_independent_results = user_independent_evaluation_with_representations(classifier)
    
    print("\nRunning user-dependent evaluation...")
    user_dependent_results = user_dependent_evaluation_with_representations(classifier)
    
    print("\nComparison of results:")
    print(f"User-independent accuracy: {user_independent_results['mean_accuracy']:.4f} ± {user_independent_results['std_accuracy']:.4f}")
    print(f"User-dependent accuracy: {user_dependent_results['mean_accuracy']:.4f} ± {user_dependent_results['std_accuracy']:.4f}")