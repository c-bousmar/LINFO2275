import argparse

import pandas as pd
from sklearn.cluster import KMeans

from datasets_utils import get_dataset_from_domain


def clustering(n_clusters, dataset):
    """
    Perform KMeans clustering on all 3D points from the dataset.

    Parameters:
    - n_clusters (int): Number of clusters to use.
    - dataset (pd.DataFrame): Gesture dataset with columns <x>, <y>, <z>, etc.

    Returns:
    - kmeans (KMeans): Trained KMeans model with centroids.
    """
    all_points = dataset[['<x>', '<y>', '<z>']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(all_points)
    return kmeans


def label_conversion(dataset, kmeans):
    """
    Append cluster labels to each 3D point in the dataset.

    Parameters:
    - dataset (pd.DataFrame): Original dataset with 3D points.
    - kmeans (KMeans): Trained KMeans model.

    Returns:
    - pd.DataFrame: Dataset with an added 'label' column indicating nearest cluster label (as a char).
    """
    coords = dataset[['<x>', '<y>', '<z>']].values
    numeric_labels = kmeans.predict(coords)

    label_chars = [chr(65 + l) for l in numeric_labels]  # A, B, C, ...
    labeled_dataset = dataset.copy()
    labeled_dataset['label'] = label_chars
    return labeled_dataset


def label_gestures(labelled_data):
    """
    Aggregate labelled 3D points into gesture-level sequences.

    Parameters:
    - labelled_data (pd.DataFrame): Data with 'label' column added per 3D point.

    Returns:
    - pd.DataFrame: One row per gesture, with sequence and metadata.
    """
    gesture_groups = labelled_data.groupby(['domain', 'subject_id', 'trial_id'])
    gestures = []
    for (domain_id, subject_id, trial_id), group in gesture_groups:
        sequence = ''.join(group.sort_values('<t>')['label'].tolist())
        gestures.append({
            'gesture': sequence,
            'subject_id': subject_id,
            'trial_id': trial_id,
            'target': group['target'].iloc[0],
            'domain': domain_id
        })
    return pd.DataFrame(gestures)


def convert_3d_to_labels(domains, verbose=False):
    """
    Converts 3D gesture coordinates from multiple domains into sequences of cluster labels,
    then saves the aggregated gesture-level sequences to a CSV file.

    This function performs vector quantization of 3D gesture data using KMeans clustering,
    assigns each point a label corresponding to its nearest centroid, and generates a string
    representation of each gesture. The output CSV contains one row per gesture, preserving
    metadata such as subject ID, trial ID, target class, and domain.

    Args:
        domains (list[int]): List of domain identifiers to load and process (e.g., [1], [1, 4]).
        verbose (bool, optional): If True, prints progress messages to stdout. Default is False.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to one gesture, with columns:
            'gesture' (label sequence), 'subject_id', 'trial_id', 'target', and 'domain'.

    Saves:
        A CSV file named `labelled_gestures_{d1-d2-...}.csv` in the ../Data/ directory,
        where d1-d2-... are the sorted domain IDs used.
    """
    dataset = None
    for domain in domains:
        if verbose:
            print(f'Loading domain {domain}...')
        if dataset is None:
            dataset = get_dataset_from_domain("./Data/dataset.csv", domain)
        else:
            dataset = pd.concat([dataset, get_dataset_from_domain("./Data/dataset.csv", domain)], ignore_index=True)

    number_of_clusters = len(domains) * 10
    if verbose:
        print(f'Number of clusters: {number_of_clusters}')
    if verbose:
        print('Clustering...')
    centroids = clustering(number_of_clusters, dataset)
    if verbose:
        print('Labeling clusters...')
    labelled_data = label_conversion(dataset, centroids)
    if verbose:
        print('Labeling gestures...')
    labelled_gestures = label_gestures(labelled_data)
    if verbose:
        print('Saving labelled gestures...')
    domain_suffix = "-".join(map(str, sorted(domains)))
    output_filename = f"./Data/labelled_gestures_{domain_suffix}.csv"
    labelled_gestures.to_csv(output_filename, index=False)
    return labelled_gestures


def main():
    parser = argparse.ArgumentParser(description="Convert 3D gesture data to labelled sequences.")
    parser.add_argument("--domains", nargs="+", type=int, default=[1],
                        help="List of domain IDs to process (default: [1])")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output with progress bars.")
    args = parser.parse_args()

    convert_3d_to_labels(args.domains, verbose=args.verbose)


if __name__ == "__main__":
    main()
