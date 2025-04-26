import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.base import clone
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from datasets_utils import get_dataset_from_domain

class DollarOne3DRecognizer:
    def __init__(self, num_points=64):
        self.num_points = num_points
        self.pca = PCA(n_components=2)
        self.templates = []
    
    def _resample(self, points):
        cumulative_dist = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        cumulative_dist = np.insert(cumulative_dist, 0, 0)
        
        if cumulative_dist[-1] == 0:
            return np.tile(points[0], (self.num_points, 1))
            
        return interp1d(cumulative_dist, points, axis=0)(np.linspace(0, cumulative_dist[-1], self.num_points))
    
    def _pca_project(self, points):
        return self.pca.transform(points)
    
    def _rotate_to_base(self, points):
        delta = points[-1] - points[0]
        angle = np.arctan2(delta[1], delta[0])
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return points @ rotation_matrix
    
    def _normalize_scale(self, points):
        min_val = np.min(points)
        max_val = np.max(points)
        return (points - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else points
    
    def _preprocess(self, points):
        resampled = self._resample(points)
        projected = self._pca_project(resampled)
        rotated = self._rotate_to_base(projected)
        return self._normalize_scale(rotated)
    
    def fit(self, X_train, y_train=None):
        all_points = np.vstack(X_train)
        self.pca.fit(all_points)
        
    def add_template(self, gesture, label):
        processed = self._preprocess(gesture)
        self.templates.append({'points': processed, 'label': label})
        
    def predict(self, X):
        predictions = []
        for gesture in X:
            pred = self.recognize(gesture)
            predictions.append(pred)
        return np.array(predictions)
    
    def fit(self, X_train, y_train):
        all_points = np.vstack([self._resample(g) for g in X_train])
        self.pca.fit(all_points)
        
        self.templates = []
        for gesture, label in zip(X_train, y_train):
            processed = self._preprocess(gesture)
            self.templates.append({
                'points': processed,
                'label': label
            })

    def recognize(self, gesture):
        candidate = self._preprocess(gesture)
        min_score = float('inf')
        best_label = "unknown" 
        for template in self.templates:
            _, _, disparity = procrustes(template['points'], candidate)
            if disparity < min_score:
                min_score = disparity
                best_label = template['label']
                
        return best_label


def plot_confusion_matrix(y_true, y_pred, labels,filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix $1 + DTW')
    plt.savefig(filename, format="pdf")

    
def evaluate_user_independent(X, y, users):
    unique_users = np.unique(users)
    accuracies = []
    y_true_all = []
    y_pred_all = []
    
    print("\nUser-Independent Evaluation:")
    for user in tqdm(unique_users, desc="Processing users"):
        train_mask = users != user
        test_mask = users == user
        
        model = DollarOne3DRecognizer()
        model.fit(X[train_mask], y[train_mask])
        
        y_pred = model.predict(X[test_mask])
        
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(y_pred)
        
        acc = accuracy_score(y[test_mask], y_pred)
        accuracies.append(acc)
    
    filename = "./Project2/src/advanced/results/features_user_independent_$1_DTW_domain1.pdf"
    plot_confusion_matrix(np.array(y_true_all), np.array(y_pred_all), np.unique(y),filename)
    
    return np.mean(accuracies), np.std(accuracies)


def evaluate_user_dependent(X, y, users):
    df = pd.DataFrame({'user': users, 'gesture': y, 'index': np.arange(len(y))})
    accuracies = []
    y_true_all = []
    y_pred_all = []
    
    print("\nUser-Dependent Evaluation:")
    for fold in tqdm(range(10), desc="Processing folds"):
        test_indices = []
        for (u, g), group in df.groupby(['user', 'gesture']):
            samples = group.index.values
            if len(samples) > fold:
                test_indices.append(samples[fold])
        
        test_mask = df.index.isin(test_indices)
        train_mask = ~test_mask
        
        model = DollarOne3DRecognizer()
        model.fit(X[train_mask], y[train_mask])
        
        y_pred = model.predict(X[test_mask])
        
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(y_pred)
        
        acc = accuracy_score(y[test_mask], y_pred)
        accuracies.append(acc)
    
    filename = f'./Project2/src/advanced/results/features_user_dependent_$1_DTW_domain1.pdf'
    plot_confusion_matrix(np.array(y_true_all), np.array(y_pred_all), np.unique(y),filename)
    
    return np.mean(accuracies), np.std(accuracies)

def load_data(file_path):
    df = get_dataset_from_domain(file_path, domain_number=1)
    grouped = df.groupby("source_file")
    gestures, labels, users = [], [], []
    
    for name, group in grouped:
        gestures.append(group[["<x>", "<y>", "<z>"]].values)
        labels.append(group["target"].iloc[0])
        users.append(group["subject_id"].iloc[0])
    
    return np.array(gestures, dtype=object), np.array(labels), np.array(users)


if __name__ == "__main__":
    X, y, users = load_data("/Users/leoncelamien/Desktop/LINFO2275/Project2/Data/dataset.csv")
    
    mean_acc, std_acc = evaluate_user_independent(X, y, users)
    print(f"\nUser-Independent → Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
    
    mean_acc, std_acc = evaluate_user_dependent(X, y, users)
    print(f"\nUser-Dependent → Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")