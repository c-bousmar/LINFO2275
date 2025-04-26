import sys
import os
sys.path.append(os.path.abspath('../'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from datasets_utils import get_dataset_from_domain

def create_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    return model


def find_best_max_length(df, percentile=95):
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    
    lengths = [len(group) for _, group in gesture_groups]
    
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    
    best_max_length = int(np.percentile(lengths, percentile))
    print(f"Suggested max_length at {percentile}% percentile: {best_max_length}")
    
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(best_max_length, color='red', linestyle='dashed', linewidth=2, label=f'{percentile}th percentile')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Gesture Sequence Lengths')
    plt.legend()
    plt.show()
    
    return best_max_length


def prepare_gesture_sequences(df, max_length=100, normalize=True):
    gesture_groups = df.groupby(['subject_id', 'target', 'trial_id'])
    sequences = []
    labels = []
    subject_ids = []
    trial_ids = []
    
    for (subject_id, target, trial_id), group in gesture_groups:
        sequence = group.sort_values('<t>')[['<x>', '<y>', '<z>']].values
        
        if normalize:
            min_vals = np.min(sequence, axis=0)
            max_vals = np.max(sequence, axis=0)
            range_vals = np.maximum(max_vals - min_vals, 1e-10)
            sequence = (sequence - min_vals) / range_vals
            
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            padding_length = max_length - len(sequence)
            if padding_length > 0:
                padding = np.zeros((padding_length, 3))
                sequence = np.vstack([sequence, padding])
        
        sequences.append(sequence)
        labels.append(target)
        subject_ids.append(subject_id)
        trial_ids.append(trial_id)
    
    sequences = np.array(sequences)
    
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    else:
        labels = np.array(labels)
    
    return sequences, labels, np.array(subject_ids), np.array(trial_ids)


def user_independent_lstm_score(domain_number=1, epochs=50, batch_size=32, verbose=True):
    # Load the dataset
    dataset_path = f"../Data/dataset.csv"
    if verbose:
        print(f"Loading dataset from {dataset_path}")
    
    df = get_dataset_from_domain(dataset_path, domain_number)
    
    # Get unique subject IDs and targets
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
        
        max_length = find_best_max_length(train_df)
        
        # Prepare the training and test sets
        X_train, y_train, _, _ = prepare_gesture_sequences(train_df, max_length=max_length)
        X_test, y_test, _, _ = prepare_gesture_sequences(test_df, max_length=max_length)
        
        # Get unique classes
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        
        # Train the model
        input_shape = (X_train.shape[1], X_train.shape[2])
        current_model = create_lstm_model(input_shape, num_classes)
        
        # Use early stopping to prevent overfitting
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            # ModelCheckpoint(f'lstm_model_domain{domain_number}.h5', save_best_only=True, monitor='val_accuracy')
        ]
        current_model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0
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
    
    # Calculate overall metrics
    mean_accuracy = np.mean(user_accuracies)
    std_accuracy = np.std(user_accuracies)
    
    # Generate confusion matrix
    unique_targets = np.unique(all_true_labels)
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=unique_targets)
    report = classification_report(all_true_labels, all_pred_labels)
    
    if verbose:
        print(f"\nUser-independent evaluation results for domain {domain_number}:")
        print(f"Mean accuracy: {mean_accuracy:.4f}")
        print(f"Standard deviation: {std_accuracy:.4f}")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_targets, yticklabels=unique_targets)
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

if __name__ == "__main__":
    results = user_independent_lstm_score(domain_number=1, epochs=50, batch_size=32)
    print(f"Mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")