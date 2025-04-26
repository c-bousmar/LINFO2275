import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.manifold import MDS
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from datasets_utils import get_dataset_from_domain
from baseline.distance_metrics import dtw_distance

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

def compute_distance_matrix(sequences, distance_function):
    n_sequences = len(sequences)
    distance_matrix = np.zeros((n_sequences, n_sequences))
    
    # Compute upper triangular part of the distance matrix
    for i in tqdm(range(n_sequences), desc="Computing distance matrix"):
        for j in range(i, n_sequences):
            distance = distance_function(sequences[i], sequences[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric
    
    return distance_matrix

def learn_representation_with_mds(distance_matrix, n_components=5):
    mds = MDS(n_components=n_components, 
              dissimilarity='precomputed',
              random_state=42,
              n_jobs=-1)
    representation = mds.fit_transform(distance_matrix)
    return representation

def user_independent_evaluation_with_representations(classifier,
                                                    domain_number=1, 
                                                    distance_function=dtw_distance,
                                                    n_components=5,
                                                    method='mds',
                                                    verbose=True):
    # Load the dataset
    if verbose:
        print(f"Loading dataset for domain {domain_number}...")
    
    df = get_dataset_from_domain("../Data/dataset.csv", domain_number)

    classifier_type = list(classifier.named_steps.keys())[1]
    
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
        print(f"Computing pairwise distances using {distance_function.__name__}...")
    
    # Compute full distance matrix
    distance_matrix = compute_distance_matrix(sequences, distance_function)
    
    if verbose:
        print(f"Learning {n_components}-dimensional representation with MDS...")
    
    # Learn representation using MDS
    representations = learn_representation_with_mds(distance_matrix, n_components)

    # Create a mapping from sequences to data
    sequence_data = list(zip(representations, labels, subject_ids))
    
    # Leave-one-user-out cross-validation
    for test_subject in tqdm(unique_subjects, desc="User cross-validation", disable=not verbose):
        # Split data
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
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} ({method.upper()}, {classifier_type})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'advanced/results/mds_{method}_{classifier_type}_domain{domain_number}.pdf', format="pdf")
        plt.close()
        
        # Visualize 2D projection
        if n_components >= 2:
            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(representations[indices, 0], representations[indices, 1], 
                           label=f'Class {label}', alpha=0.7)
            
            plt.title(f'MDS Representation of Gestures (Domain {domain_number})')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'advanced/results/mds_visualization_domain{domain_number}.png')
            plt.show()
    
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
    
if __name__ == "__main__":
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=1000, multi_class='multinomial'))
        # ('svm', SVC(C=10, gamma='scale', probability=True))
        # ('nn', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, early_stopping=True))
    ])
    results = user_independent_evaluation_with_representations(classifier)