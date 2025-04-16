import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def get_dataset_from_domain(dataset_path, domain_number):
    """
    Load and filter dataset by domain, with appropriate target type casting.

    This function reads the dataset from the specified path, filters it based on
    the provided domain number, and ensures the 'target' column is correctly
    cast to the expected data type depending on the domain:
        - Domain 1: targets are integers
        - Domain 4: targets are strings

    Parameters
    ----------
    dataset_path : str
        Path to the CSV dataset file.
    domain_number : int
        Domain identifier to filter the dataset (1 or 4 supported).

    Returns
    -------
    pandas.DataFrame or None
        Filtered DataFrame with correct 'target' column types if domain is valid;
        otherwise, returns None and prints an error.

    Notes
    -----
    - Only domain 1 and domain 4 are currently supported.
    - If an unsupported domain is provided, the function will return None.
    """
    df = pd.read_csv(dataset_path)
    df = df[df['domain'] == domain_number]
    if (domain_number == 1):
        df['target'] = df['target'].astype(int)
    elif (domain_number == 4):
        df['target'] = df['target'].astype(str)
    else:
        print("Error - Only Domain 1 and 4 Available for now.")
        return None
    return df

def user_independent_score(model, domain_number=1, verbose=True):
    """
    Evaluate model performance using leave-one-user-out cross-validation.
    
    This function implements the user-independent setting where:
    1. One user is left out as test set
    2. The model is trained on the remaining 9 users
    3. Evaluation is done on the left-out user
    4. The process is repeated for all users
    
    Parameters:
    -----------
    model : object
        Model object with fit() and predict() methods
    domain_number : int, default=1
        Domain number (1 or 4) to specify which dataset to use
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
    # Load the dataset
    dataset_path = f"Data/dataset.csv"
    if verbose:
        print(f"Loading dataset from {dataset_path}")
    
    df = get_dataset_from_domain(dataset_path, domain_number)
    
    # Get unique subject IDs and targets
    subject_ids = sorted(df['subject_id'].unique())
    targets = sorted(df['target'].unique())
    
    # Initialize results storage
    user_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting leave-one-user-out cross-validation for domain {domain_number}")
        print(f"Total users: {len(subject_ids)}")
        print(f"Total targets/classes: {len(targets)}")
    
    # Iterate through each user
    for test_subject in tqdm(subject_ids, desc="Users", disable=not verbose):
        # Split data into training and test sets
        train_df = df[df['subject_id'] != test_subject]
        test_df = df[df['subject_id'] == test_subject]
        
        # Prepare the training and test sets
        # TODO - To Adapt (Here keep coordinates only)
        X_train = train_df.drop(['subject_id', 'target', 'trial_id', 'source_file', 'domain'], axis=1, errors='ignore')
        y_train = train_df['target']
        
        X_test = test_df.drop(['subject_id', 'target', 'trial_id', 'source_file', 'domain'], axis=1, errors='ignore')
        y_test = test_df['target']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
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
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=targets)
    
    if verbose:
        print(f"\nUser-independent evaluation results for domain {domain_number}:")
        print(f"Mean accuracy: {mean_accuracy:.4f}")
        print(f"Standard deviation: {std_accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=targets, yticklabels=targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (User-Independent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    # Return results
    return {
        'user_accuracies': user_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }

def user_dependent_score(model, domain_number=1, verbose=True):
    """
    Evaluate model performance using leave-one-gesture-sample-out cross-validation.
    
    This function implements the user-dependent setting where:
    1. For each user, one sample of each gesture class is held out as test
    2. The model is trained on the remaining 9 samples of each gesture for all users
    3. Evaluation is done on the held-out samples
    4. The process is repeated for all 10 trials
    
    Parameters:
    -----------
    model : object
        Model object with fit() and predict() methods
    domain_number : int
        Domain number (1 or 4) to specify which dataset to use
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
   # Load the dataset
    dataset_path = f"Data/dataset.csv"
    if verbose:
        print(f"Loading dataset from {dataset_path}")
    
    df = get_dataset_from_domain(dataset_path, domain_number)
    
    # Get unique subject IDs, targets, and trials
    subject_ids = sorted(df['subject_id'].unique())
    targets = sorted(df['target'].unique())
    trial_ids = sorted(df['trial_id'].unique())
    
    # Initialize results storage
    trial_accuracies = []
    all_true_labels = []
    all_pred_labels = []
    
    if verbose:
        print(f"Starting leave-one-gesture-sample-out cross-validation for domain {domain_number}")
        print(f"Total users: {len(subject_ids)}")
        print(f"Total targets/classes: {len(targets)}")
        print(f"Total trials: {len(trial_ids)}")
    
    # Iterate through each trial for cross-validation
    for test_trial in tqdm(trial_ids, desc="Trials", disable=not verbose):
        # For each trial, hold out one sample of each gesture for each user
        test_indices = []
        train_indices = []
        
        for subject in subject_ids:
            for target in targets:
                # Get all samples for this subject-target combination
                subject_target_indices = df[
                    (df['subject_id'] == subject) & 
                    (df['target'] == target)
                ].index
                
                # Get test samples (samples with current trial_id)
                test_sample_indices = df[
                    (df['subject_id'] == subject) & 
                    (df['target'] == target) & 
                    (df['trial_id'] == test_trial)
                ].index
                
                # Get train samples (all other samples)
                train_sample_indices = df[
                    (df['subject_id'] == subject) & 
                    (df['target'] == target) & 
                    (df['trial_id'] != test_trial)
                ].index
                
                test_indices.extend(test_sample_indices)
                train_indices.extend(train_sample_indices)
        
        # Split data into training and test sets
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]
        
        # Prepare the training and test sets
        # TODO - To Adapt (Here keep coordinates only)
        X_train = train_df.drop(['subject_id', 'target', 'trial_id', 'source_file', 'domain'], axis=1, errors='ignore')
        y_train = train_df['target']
        
        X_test = test_df.drop(['subject_id', 'target', 'trial_id', 'source_file', 'domain'], axis=1, errors='ignore')
        y_test = test_df['target']
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
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
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=targets)
    
    if verbose:
        print(f"\nUser-dependent evaluation results for domain {domain_number}:")
        print(f"Mean accuracy: {mean_accuracy:.4f}")
        print(f"Standard deviation: {std_accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=targets, yticklabels=targets)
        plt.title(f'Confusion Matrix - Domain {domain_number} (User-Dependent)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    # Return results
    return {
        'trial_accuracies': trial_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrix': conf_matrix
    }