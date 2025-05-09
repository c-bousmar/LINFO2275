import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class GestureRecognitionEvaluator:
    """
    A unified evaluator for gesture recognition models that handles both 
    user-dependent and user-independent evaluation protocols.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        verbose : bool
            Whether to display progress and results.
        """
        self.verbose = verbose
    
    def _prepare_sequences(self, model,df, normalize=True, use_string_repr=False):
        """
        Prepare sequences from dataframe, grouping by subject, target, and trial.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing gesture data.
        normalize : bool
            Whether to normalize the sequences.
            
        Returns:
        --------
        tuple
            (sequences, labels, subject_ids, trial_ids)
        """
        if use_string_repr:            
            return df["gesture"].tolist(), df["target"].tolist(), df["subject_id"].tolist(), df["trial_id"].tolist()
    
        # Group the dataset by several features: subject_id, target and trial_id
        gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
        sequences, labels, subject_ids, trial_ids = [], [], [], []
        
        for (subject_id, target, trial_id), group in gesture_groups:
            # Sort the sequences by time and get the coordinates only
            if(model.__class__.__name__ == "DollarOne3DGestureRecognizer"):
                sequence = group.sort_values('<t>')[['<x>', '<y>']].values
            else:
                sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
            
            # Normalize if requested
            if normalize:
                min_vals = np.min(sequence, axis=0)
                max_vals = np.max(sequence, axis=0)
                range_vals = np.maximum(max_vals - min_vals, 1e-10)  # Avoid division by zero
                sequence = (sequence - min_vals) / range_vals
            
            sequences.append(sequence)
            labels.append(target)
            subject_ids.append(subject_id)
            trial_ids.append(trial_id)
        
        return sequences, labels, subject_ids, trial_ids


    def evaluate(self, model, df, evaluation_type='user-independent', normalize=True, use_string_repr=False, n_folds=10):
        """
        Evaluate a gesture recognition model.
        
        Parameters:
        -----------
        model : object
            The model to evaluate. Must implement fit() and predict() methods.
        df : pandas.DataFrame
            DataFrame containing gesture data.
        evaluation_type : str
            Either 'user-independent' (leave-one-user-out) or 'user-dependent' (leave-one-trial-out).
        n_folds : int
            Number of folds for k-fold cross-validation (used in user-dependent evaluation).
        normalize : boolean
            Whether to normalize the sequences.
        Returns:
        --------
        dict
            Results including accuracies, mean, std, and confusion matrix.
        """
        if self.verbose:
            print(f"\nPerforming {evaluation_type} evaluation:")
        
        # Prepare sequences
        sequences, labels, subject_ids, trial_ids = self._prepare_sequences(model,df, normalize, use_string_repr)
        
        if evaluation_type == 'user-independent':
            return self._evaluate_user_independent(model, sequences, labels, subject_ids)
        elif evaluation_type == 'user-dependent':
            return self._evaluate_user_dependent(model, sequences, labels, subject_ids, trial_ids, n_folds)
        else:
            raise ValueError("evaluation_type must be either 'user-independent' or 'user-dependent'")
    
    def _evaluate_user_independent(self, model, sequences, labels, subject_ids):
        """
        Perform leave-one-user-out cross-validation.
        
        Parameters:
        -----------
        model : object
            The model to evaluate. Must implement fit() and predict() methods.
        sequences : list
            List of gesture sequences.
        labels : list
            List of gesture labels.
        subject_ids : list
            List of subject identifiers.
            
        Returns:
        --------
        dict
            Results including user accuracies, mean, std, and confusion matrix.
        """
        sequences = np.array(sequences, dtype=object)
        labels = np.array(labels)
        subject_ids = np.array(subject_ids)
        
        unique_subjects = np.unique(subject_ids)
        unique_labels = np.unique(labels)
        
        if self.verbose:
            print(f"Total gestures: {len(sequences)}")
            print(f"Total users: {len(unique_subjects)}")
            print(f"Total classes: {len(unique_labels)}")
        
        # Initialize results
        user_accuracies = []
        all_true_labels = []
        all_pred_labels = []
        
        # Leave-one-user-out cross-validation
        for test_subject in tqdm(unique_subjects, desc="Processing users", disable=not self.verbose):
            # Split data
            train_mask = subject_ids != test_subject
            test_mask = subject_ids == test_subject
            
            # Train model
            model.fit(sequences[train_mask], labels[train_mask])
            
            # Make predictions
            y_pred = model.predict(sequences[test_mask])
            
            # Calculate accuracy
            acc = accuracy_score(labels[test_mask], y_pred)
            user_accuracies.append(acc)
            
            # Store for confusion matrix
            all_true_labels.extend(labels[test_mask])
            all_pred_labels.extend(y_pred)
            
            if self.verbose:
                print(f"User {test_subject} accuracy: {acc:.4f}")
        
        # Calculate overall metrics
        mean_acc = np.mean(user_accuracies)
        std_acc = np.std(user_accuracies)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)
        
        if self.verbose:
            print(f"\nUser-Independent evaluation results:")
            print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_true_labels, all_pred_labels, labels=unique_labels))
        
        return {
            'evaluation_type': 'user-independent',
            'accuracies': user_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'confusion_matrix': cm,
            'true_labels': all_true_labels,
            'pred_labels': all_pred_labels,
            'unique_labels': unique_labels
        }
    
    def _evaluate_user_dependent(self, model, sequences, labels, subject_ids, trial_ids, n_folds=10):
        """
        Perform user-dependent evaluation (leave-one-trial-out or k-fold).
        
        Parameters:
        -----------
        model : object
            The model to evaluate. Must implement fit() and predict() methods.
        sequences : list
            List of gesture sequences.
        labels : list
            List of gesture labels.
        subject_ids : list
            List of subject identifiers.
        trial_ids : list
            List of trial identifiers.
        n_folds : int
            Number of folds for k-fold cross-validation.
            
        Returns:
        --------
        dict
            Results including fold accuracies, mean, std, and confusion matrix.
        """
        sequences = np.array(sequences, dtype=object)
        labels = np.array(labels)
        subject_ids = np.array(subject_ids)
        trial_ids = np.array(trial_ids)
        
        unique_labels = np.unique(labels)
        
        # Create a DataFrame to help with k-fold splitting
        df_splits = pd.DataFrame({
            'label': labels,
            'subject': subject_ids,
            'trial': trial_ids,
            'idx': np.arange(len(sequences))
        })
        
        # Initialize results
        fold_accuracies = []
        all_true_labels = []
        all_pred_labels = []
        
        if len(np.unique(trial_ids)) >= n_folds:
            # Use trials for splits if there are enough
            for fold, test_trial in enumerate(tqdm(np.unique(trial_ids)[:n_folds], desc="Processing trials", disable=not self.verbose)):
            # for fold, test_trial in enumerate(np.unique(trial_ids)[:n_folds]):
                if self.verbose:
                    print(f"Fold {fold+1}/{n_folds} - Testing on trial {test_trial}")
                
                test_indices = df_splits[df_splits['trial'] == test_trial].idx.values
                test_mask = np.zeros(len(sequences), dtype=bool)
                test_mask[test_indices] = True
                train_mask = ~test_mask
                
                # Train and predict
                model.fit(sequences[train_mask], labels[train_mask])
                y_pred = model.predict(sequences[test_mask])
                
                # Calculate accuracy
                acc = accuracy_score(labels[test_mask], y_pred)
                fold_accuracies.append(acc)
                
                # Store for confusion matrix
                all_true_labels.extend(labels[test_mask])
                all_pred_labels.extend(y_pred)
                
                if self.verbose:
                    print(f"Trial {test_trial} accuracy: {acc:.4f}")
        else:
            # Use stratified k-fold
            for fold in range(n_folds):
                if self.verbose:
                    print(f"Fold {fold+1}/{n_folds}")
                
                test_indices = []
                # Stratified sampling for each user-gesture combination
                for _, group in df_splits.groupby(['subject', 'label']):
                    indices = group.idx.values
                    if len(indices) > fold:
                        test_indices.append(indices[fold % len(indices)])
                
                test_mask = np.zeros(len(sequences), dtype=bool)
                test_mask[test_indices] = True
                train_mask = ~test_mask
                
                # Clone the model
                try:
                    from sklearn.base import clone
                    current_model = clone(model)
                except:
                    current_model = model
                
                # Train and predict
                current_model.fit(sequences[train_mask], labels[train_mask])
                y_pred = current_model.predict(sequences[test_mask])
                
                # Calculate accuracy
                acc = accuracy_score(labels[test_mask], y_pred)
                fold_accuracies.append(acc)
                
                # Store for confusion matrix
                all_true_labels.extend(labels[test_mask])
                all_pred_labels.extend(y_pred)
                
                if self.verbose:
                    print(f"Fold {fold+1} accuracy: {acc:.4f}")
        
        # Calculate overall metrics
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_labels)
        
        if self.verbose:
            print(f"\nUser-Dependent evaluation results:")
            print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(all_true_labels, all_pred_labels, labels=unique_labels))
        
        return {
            'evaluation_type': 'user-dependent',
            'accuracies': fold_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'confusion_matrix': cm,
            'true_labels': all_true_labels,
            'pred_labels': all_pred_labels,
            'unique_labels': unique_labels
        }
    
    def save_results_to_csv(self, results, output_file):
        """
        Save evaluation results to a CSV file.
        
        @param results: Results dictionary from evaluator
        @param output_file: Path to save the CSV file
        """
        # Create dictionary for dataframe
        data = {
            'fold': [],
            'accuracy': [],
            'type': []
        }
        
        # Add individual fold accuracies
        for i, acc in enumerate(results['accuracies']):
            data['fold'].append(i+1)
            data['accuracy'].append(acc)
            data['type'].append('fold')
        
        # Add mean accuracy
        data['fold'].append('mean')
        data['accuracy'].append(results['mean_accuracy'])
        data['type'].append('summary')
        
        # Add std accuracy
        data['fold'].append('std')
        data['accuracy'].append(results['std_accuracy'])
        data['type'].append('summary')
        
        # Create dataframe and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        # Also save confusion matrix to a separate file
        cm_file = output_file.replace('.csv', '_confusion_matrix.csv')
        pd.DataFrame(
            results['confusion_matrix'], 
            index=results['unique_labels'],
            columns=results['unique_labels']
        ).to_csv(cm_file)
        
        print(f"Results saved to {output_file}")
        print(f"Confusion matrix saved to {cm_file}")
            
    def plot_confusion_matrix(self, results, title=None, filename=None):
        """
        Plot a confusion matrix from the evaluation results.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from evaluate method.
        title : str
            Title for the plot.
        filename : str
            If provided, save the plot to this file.
        """
        cm = results['confusion_matrix']
        labels = results['unique_labels']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        
        if title is None:
            title = f'Confusion Matrix ({results["evaluation_type"]})'
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, format="pdf")
            if self.verbose:
                print(f"Confusion matrix saved to {filename}")
        
        plt.show()
    
    def compare_models(self, models_dict, df, evaluation_types=None, n_folds=10, output_csv="advanced_comparison.csv"):
        """
        Compare multiple models with both evaluation methods and save results to CSV.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary mapping model names to model objects.
        df : pandas.DataFrame
            DataFrame containing gesture data.
        evaluation_types : list
            List of evaluation types to use. Default: ['user-independent', 'user-dependent']
        n_folds : int
            Number of folds for user-dependent evaluation.
        output_csv : str, optional
            Path to save results CSV file. If None, results are not saved.
            
        Returns:
        --------
        dict
            Results for each model and evaluation type.
        """
        if evaluation_types is None:
            evaluation_types = ['user-independent', 'user-dependent']
        
        results = {}
        
        # Create a list to store rows for CSV
        csv_rows = []
        
        for model_name, model in models_dict.items():
            results[model_name] = {}
            for eval_type in evaluation_types:
                if self.verbose:
                    print(f"\n{'-'*50}")
                    print(f"Evaluating {model_name} with {eval_type} protocol")
                    print(f"{'-'*50}")
                
                eval_results = self.evaluate(
                    model, df, evaluation_type=eval_type, n_folds=n_folds
                )
                
                results[model_name][eval_type] = eval_results
                
                # Add data to CSV rows
                for fold_idx, fold_acc in enumerate(eval_results.get('fold_accuracies', [])):
                    csv_rows.append({
                        'model': model_name,
                        'evaluation_type': eval_type,
                        'fold': fold_idx,
                        'accuracy': fold_acc
                    })
                
                # Add mean results
                csv_rows.append({
                    'model': model_name,
                    'evaluation_type': eval_type,
                    'fold': 'mean',
                    'accuracy': eval_results['mean_accuracy']
                })
                
                # Add std results
                csv_rows.append({
                    'model': model_name,
                    'evaluation_type': eval_type,
                    'fold': 'std',
                    'accuracy': eval_results['std_accuracy']
                })
        
        # Print summary
        if self.verbose:
            print("\n" + "="*80)
            print("SUMMARY OF RESULTS")
            print("="*80)
            for model_name in models_dict.keys():
                print(f"\n{model_name}:")
                for eval_type in evaluation_types:
                    mean_acc = results[model_name][eval_type]['mean_accuracy']
                    std_acc = results[model_name][eval_type]['std_accuracy']
                    print(f"  {eval_type}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Save results to CSV if output_csv is provided
        if output_csv is not None:
            import pandas as pd
            results_df = pd.DataFrame(csv_rows)
            results_df.to_csv(output_csv, index=False)
            
            if self.verbose:
                print(f"\nResults saved to {output_csv}")
        elif self.verbose:
            # If output_csv is None but we want to save results anyway (as requested)
            import pandas as pd
            import os
            from datetime import datetime
            
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_csv = f"model_comparison_results_{timestamp}.csv"
            
            results_df = pd.DataFrame(csv_rows)
            results_df.to_csv(default_csv, index=False)
            
            print(f"\nResults saved to {default_csv}")
        
        return results

    def get_dataset_from_domain(self, file_path, domain_number):
        """
        Utility function to get a dataset for a specific domain.
        This is a placeholder - actual implementation depends on your dataset structure.
        
        Parameters:
        -----------
        file_path : str
            Path to the dataset file.
        domain_number : int
            Domain number to filter.
            
        Returns:
        --------
        pandas.DataFrame
            Filtered dataset.
        """
        # This is a placeholder - implement according to your needs
        df = pd.read_csv(file_path)
        # Assume there's a column that indicates the domain
        return df[df['domain'] == domain_number]