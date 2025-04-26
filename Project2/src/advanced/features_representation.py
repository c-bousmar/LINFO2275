import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from datasets_utils import get_dataset_from_domain

def prepare_gesture_sequences(df, normalize=True):
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    sequences = []
    labels = []
    subject_ids = []
    trial_ids = []
    
    for (subject_id, target, trial_id), group in gesture_groups:
        # Extract and sort by time
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
    # Basic statistical features
    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)
    
    # Range of motion
    range_of_motion = max_vals - min_vals
    
    # Velocity features (first derivative)
    if len(sequence) > 1:
        velocity = np.diff(sequence, axis=0)
        mean_velocity = np.mean(velocity, axis=0)
        std_velocity = np.std(velocity, axis=0)
        max_velocity = np.max(np.abs(velocity), axis=0)
    else:
        mean_velocity = np.zeros(3)
        std_velocity = np.zeros(3)
        max_velocity = np.zeros(3)
    
    # Acceleration features (second derivative)
    if len(sequence) > 2:
        acceleration = np.diff(velocity, axis=0)
        mean_acceleration = np.mean(acceleration, axis=0)
        std_acceleration = np.std(acceleration, axis=0)
        max_acceleration = np.max(np.abs(acceleration), axis=0)
    else:
        mean_acceleration = np.zeros(3)
        std_acceleration = np.zeros(3)
        max_acceleration = np.zeros(3)
    
    # Gesture length and duration
    path_length = np.sum(np.sqrt(np.sum(np.diff(sequence, axis=0)**2, axis=1)))
    
    # Combine all features
    features = np.concatenate([
        mean, std, min_vals, max_vals, range_of_motion,
        mean_velocity, std_velocity, max_velocity,
        mean_acceleration, std_acceleration, max_acceleration,
        [path_length]
    ])
    
    return features

def extract_features_from_all_gestures(sequences):
    all_features = []
    for sequence in sequences:
        features = extract_features_from_gesture(sequence)
        all_features.append(features)
    return np.array(all_features)

def user_independent_evaluation_with_representations(classifier, domain_number=1, verbose=True):
    # Load the dataset
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
    
    classifier_type = list(classifier.named_steps.keys())[1]
    
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
        
        X_train = np.array([item[0] for item in train_data])
        y_train = np.array([item[1] for item in train_data])
        X_test = np.array([item[0] for item in test_data])
        y_test = np.array([item[1] for item in test_data])
        
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
    
if __name__ == '__main__':
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=1000, multi_class='multinomial'))
        # ('svm', SVC(C=10, gamma='scale', probability=True))
        # ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True))
    ])
    user_independent_evaluation_with_representations(classifier)