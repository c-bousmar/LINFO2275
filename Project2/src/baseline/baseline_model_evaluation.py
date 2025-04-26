import sys
import os
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from distance_metrics import dtw_distance, edit_distance, lcs_distance
from KNN_Classifier import KNN_Classifier

from datasets_utils import get_dataset_from_domain

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

def user_independent_score_knn(model, domain_number=1, distance_type="dtw", verbose=True):
    """
    Evaluate KNN model performance using leave-one-user-out cross-validation.
    
    This function implements the user-independent setting where:
    1. One user is left out as test set
    2. The model is trained on the remaining 9 users
    3. Evaluation is done on the left-out user
    4. The process is repeated for all users
    
    Parameters:
    -----------
    model : object
        KNN model object with fit() and predict() methods
    domain_number : int, default=1
        Domain number (1 or 4) to specify which dataset to use
    distance_type : str, default="dtw"
        Type of distance used: "dtw", "edit", or "lcs"
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'user_accuracies': List of accuracies for each user
        - 'mean_accuracy': Mean accuracy across all users
        - 'std_accuracy': Standard deviation of accuracies
        - 'confusion_matrix': Confusion matrix for all predictions
    """
    if verbose:
        print(f"Evaluating KNN with {distance_type.upper()} distance on domain {domain_number}")
    
    # Load data based on distance type
    if distance_type in ["edit", "lcs"]:
        # For Edit Distance or LCS, use string sequences from labelled gestures
        dataset_path = f"../Data/labelled_gestures_{domain_number}.csv"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}. Run label_conversion.py first.")
            
        if verbose:
            print(f"Loading string-based sequences from {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        gestures = df["gesture"].tolist()
        targets = df["target"].tolist()
        subject_ids = df["subject_id"].tolist()
    else:
        # For DTW, use raw 3D coordinates
        dataset_path = f"../Data/dataset.csv"
        
        if verbose:
            print(f"Loading raw gesture data from {dataset_path}")
            
        df = get_dataset_from_domain(dataset_path, domain_number)
        
        # Group data by gesture
        gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
        gestures = []
        targets = []
        subject_ids = []
        
        for (subject_id, target, trial_id), group in gesture_groups:
            # Sort by time and get coordinates
            sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
            gestures.append(sequence)
            targets.append(target)
            subject_ids.append(subject_id)
    
    # Get unique subject IDs and classes
    unique_subjects = sorted(set(subject_ids))
    unique_targets = sorted(set(targets))
    
    # Initialize results storage
    user_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting leave-one-user-out cross-validation")
        print(f"Total users: {len(unique_subjects)}")
        print(f"Total targets/classes: {len(unique_targets)}")
    
    # Iterate through each user
    for test_subject in tqdm(unique_subjects, desc="Users", disable=not verbose):
        # Split data into training and test sets
        train_indices = []
        test_indices = []
        
        for i, subject in enumerate(subject_ids):
            if subject == test_subject:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        # Prepare training and test sets
        X_train = [gestures[i] for i in train_indices]
        y_train = [targets[i] for i in train_indices]
        X_test = [gestures[i] for i in test_indices]
        y_test = [targets[i] for i in test_indices]
        
        # Train the model
        if verbose:
            print(f"Training KNN model for user {test_subject}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        if verbose:
            print(f"Making predictions for user {test_subject}...")
        y_pred = model.predict(X_test, verbose=(verbose and len(X_test) > 20))
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        user_accuracies.append(accuracy)
        
        # Store true and predicted labels for confusion matrix
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
        print(f"Mean accuracy: {mean_accuracy:.4f}")
        print(f"Standard deviation: {std_accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} ({distance_type.upper()}, User-Independent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/knn_{distance_type}_user_independent_domain{domain_number}.png')
        plt.show()
    
    # Return results
    return {
        'user_accuracies': user_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }

def user_dependent_score_knn(model, domain_number=1, distance_type="dtw", verbose=True):
    """
    Evaluate KNN model performance using leave-one-gesture-sample-out cross-validation.
    
    This function implements the user-dependent setting where:
    1. For each user, one sample of each gesture class is held out as test
    2. The model is trained on the remaining samples of each gesture
    3. Evaluation is done on the held-out samples
    4. The process is repeated for all trials
    
    Parameters:
    -----------
    model : object
        KNN model object with fit() and predict_user_dependent() methods
    domain_number : int, default=1
        Domain number (1 or 4) to specify which dataset to use
    distance_type : str, default="dtw"
        Type of distance used: "dtw", "edit", or "lcs"
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'trial_accuracies': List of accuracies for each trial
        - 'mean_accuracy': Mean accuracy across all trials
        - 'std_accuracy': Standard deviation of accuracies
        - 'confusion_matrix': Confusion matrix for all predictions
    """
    if verbose:
        print(f"Evaluating KNN with {distance_type.upper()} distance on domain {domain_number}")
    
    # Load data based on distance type
    if distance_type in ["edit", "lcs"]:
        # For Edit Distance or LCS, use string sequences from labelled gestures
        dataset_path = f"../Data/labelled_gestures_{domain_number}.csv"
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}. Run label_conversion.py first.")
            
        if verbose:
            print(f"Loading string-based sequences from {dataset_path}")
            
        df = pd.read_csv(dataset_path)
        gestures = df["gesture"].tolist()
        targets = df["target"].tolist()
        subject_ids = df["subject_id"].tolist()
        trial_ids = df["trial_id"].tolist()
    else:
        # For DTW, use raw 3D coordinates
        dataset_path = f"../Data/dataset.csv"
        
        if verbose:
            print(f"Loading raw gesture data from {dataset_path}")
            
        df = get_dataset_from_domain(dataset_path, domain_number)
        
        # Group data by gesture
        gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
        gestures = []
        targets = []
        subject_ids = []
        trial_ids = []
        
        for (subject_id, target, trial_id), group in gesture_groups:
            # Sort by time and get coordinates
            sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
            gestures.append(sequence)
            targets.append(target)
            subject_ids.append(subject_id)
            trial_ids.append(trial_id)
    
    # Get unique values
    unique_trials = sorted(set(trial_ids))
    unique_targets = sorted(set(targets))
    
    # Initialize results storage
    trial_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting leave-one-gesture-sample-out cross-validation")
        print(f"Total trials: {len(unique_trials)}")
        print(f"Total targets/classes: {len(unique_targets)}")
    
    # Iterate through each trial
    for test_trial in tqdm(unique_trials, desc="Trials", disable=not verbose):
        # Split data into training and test sets
        train_indices = []
        test_indices = []
        
        for i, trial in enumerate(trial_ids):
            if trial == test_trial:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        # Prepare training and test sets
        X_train = [gestures[i] for i in train_indices]
        y_train = [targets[i] for i in train_indices]
        subject_train = [subject_ids[i] for i in train_indices]
        
        X_test = [gestures[i] for i in test_indices]
        y_test = [targets[i] for i in test_indices]
        subject_test = [subject_ids[i] for i in test_indices]
        
        # Train the model
        if verbose:
            print(f"Training KNN model for trial {test_trial}...")
        model.fit(X_train, y_train, subject_info=subject_train)
        
        # Make predictions - using user-dependent mode
        if verbose:
            print(f"Making predictions for trial {test_trial}...")
        
        if hasattr(model, 'predict_user_dependent'):
            y_pred = model.predict_user_dependent(X_test, subject_test, 
                                                verbose=(verbose and len(X_test) > 20))
        else:
            # Fall back to regular predict if predict_user_dependent is not available
            y_pred = model.predict(X_test, verbose=(verbose and len(X_test) > 20))
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        trial_accuracies.append(accuracy)
        
        # Store true and predicted labels for confusion matrix
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
        print(f"Mean accuracy: {mean_accuracy:.4f}")
        print(f"Standard deviation: {std_accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_targets, yticklabels=unique_targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} ({distance_type.upper()}, User-Dependent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/knn_{distance_type}_user_dependent_domain{domain_number}.png')
        plt.show()
    
    # Return results
    return {
        'trial_accuracies': trial_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }

def evaluate_all_knn_baselines(domain_number=1, k_values=[1, 3, 5], distance_names=["dtw", "edit", "lcs"], verbose=True):
    """
    Evaluate all KNN baseline methods on both user-independent and user-dependent settings.
    
    Parameters:
    -----------
    domain_number : int, default=1
        Domain number (1 or 4) to specify which dataset to use
    verbose : bool, default=True
        Whether to print progress and results
        
    Returns:
    --------
    dict
        Dictionary containing results for all methods and settings
    """
    results = {}
    
    # Test all distance metrics
    distance_metrics = []
    if "dtw" in distance_names:
        distance_metrics.append({"name": "dtw", "function": dtw_distance})
    elif "edit" in distance_names:
        distance_metrics.append({"name": "edit", "function": edit_distance})
    elif "lcs" in distance_names:
        distance_metrics.append({"name": "lcs", "function": lcs_distance})
    
    for distance in distance_metrics:
        distance_name = distance["name"]
        distance_function = distance["function"]
        
        for k in k_values:
            # Create KNN model
            knn_model = KNN_Classifier(k=k, distance_function=distance_function)
            
            # Model identifier
            model_id = f"knn_{distance_name}_k{k}"
            
            if verbose:
                print(f"\n===== Evaluating {model_id.upper()} =====")
            
            # User-independent evaluation
            if verbose:
                print("\n--- User-Independent Setting ---")
            indep_results = user_independent_score_knn(
                knn_model, domain_number=domain_number, 
                distance_type=distance_name, verbose=verbose
            )
            
            # User-dependent evaluation
            if verbose:
                print("\n--- User-Dependent Setting ---")
            dep_results = user_dependent_score_knn(
                knn_model, domain_number=domain_number, 
                distance_type=distance_name, verbose=verbose
            )
            
            # Store results
            results[f"{model_id}_independent"] = indep_results
            results[f"{model_id}_dependent"] = dep_results
    
    # Summarize results
    if verbose:
        print("\n===== Summary of Results =====")
        print("\nUser-Independent Setting:")
        for k in k_values:
            for distance in distance_metrics:
                model_id = f"knn_{distance['name']}_k{k}_independent"
                mean_acc = results[model_id]['mean_accuracy']
                std_acc = results[model_id]['std_accuracy']
                print(f"{model_id}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        print("\nUser-Dependent Setting:")
        for k in k_values:
            for distance in distance_metrics:
                model_id = f"knn_{distance['name']}_k{k}_dependent"
                mean_acc = results[model_id]['mean_accuracy']
                std_acc = results[model_id]['std_accuracy']
                print(f"{model_id}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Plot comparison
        plot_baseline_comparison(results, domain_number)
    
    return results

def plot_baseline_comparison(results, domain_number):
    """
    Plot comparison of different baseline methods.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing evaluation results
    domain_number : int
        Domain number for the title
    """
    # Extract independent and dependent results
    independent_models = [k for k in results.keys() if k.endswith('_independent')]
    dependent_models = [k for k in results.keys() if k.endswith('_dependent')]
    
    # Prepare data for plot
    model_names = [m.replace('_independent', '') for m in independent_models]
    independent_accs = [results[m]['mean_accuracy'] for m in independent_models]
    independent_stds = [results[m]['std_accuracy'] for m in independent_models]
    dependent_accs = [results[d]['mean_accuracy'] for d in dependent_models]
    dependent_stds = [results[d]['std_accuracy'] for d in dependent_models]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width / 2, independent_accs, width, yerr=independent_stds, 
            label='User-Independent', alpha=0.7, capsize=5)
    plt.bar(x + width / 2, dependent_accs, width, yerr=dependent_stds, 
            label='User-Dependent', alpha=0.7, capsize=5)
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(f'Comparison of KNN Baseline Methods - Domain {domain_number}')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/baseline_comparison_domain{domain_number}.png')
    plt.show()

if __name__ == "__main__":
    results = evaluate_all_knn_baselines(domain_number=1,
                                        k_values=[1, 3, 5],
                                        distance_names=["dtw", "edit", "lcs"],
                                        domain_number=1)