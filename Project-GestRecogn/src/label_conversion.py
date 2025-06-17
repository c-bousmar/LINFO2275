import argparse

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut

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


def convert_3d_to_labels(domains, verbose=False, experimental=True):
    """
    Converts 3D gesture coordinates into sequences of cluster-based labels and saves them as gesture-level sequences.

    This function supports two operational modes:

    - Experimental (default): Clusters all 3D points in the dataset using KMeans once (unsupervised preprocessing)
      and converts every gesture into a discrete sequence of cluster labels.

    - Rigorous: Performs user-independent cross-validation (leave-one-user-out), where clustering is done only
      on the training portion of each fold. The resulting centroids are used to label both train and test data
      without any data leakage. This is a more scientifically robust procedure that simulates a realistic evaluation
      of generalisation to unseen users.

    The resulting label sequences are grouped per gesture (by subject, trial, and domain) and saved into a CSV file.

    Args:
        domains (list[int]): List of domain identifiers to load from the dataset (e.g., [1], [1, 4]).
        verbose (bool, optional): If True, displays progress messages for each step. Default is False.
        experimental (bool, optional): If True, runs the simplified clustering method. If False, uses rigorous
            cross-validation-based clustering. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with one row per gesture, containing:
            - 'gesture': the sequence of cluster labels (as a string),
            - 'subject_id': ID of the user who performed the gesture,
            - 'trial_id': trial index of the gesture,
            - 'target': class label of the gesture (as a string),
            - 'domain': domain from which the gesture comes.

    Saves:
        CSV file in `./Data/` named:
        - `labelled_gestures_{d1-d2-...}_experimental.csv` in experimental mode,
        - `labelled_gestures_{d1-d2-...}.csv` in rigorous mode,
        where d1-d2-... are the sorted domain IDs.
    """
    dataset = None
    for domain in domains:
        if verbose:
            print(f'Loading domain {domain}...')
        domain_data = get_dataset_from_domain("./Data/dataset.csv", domain)
        domain_data['target'] = domain_data['target'].astype(str)
        if dataset is None:
            dataset = domain_data
        else:
            dataset = pd.concat([dataset, domain_data], ignore_index=True)

    domain_suffix = "-".join(map(str, sorted(domains)))
    number_of_clusters = len(domains) * 10
    if verbose:
        print(f'Number of clusters: {number_of_clusters}')

    if experimental:
        if verbose:
            print('Experimental clustering...')
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
        output_filename = f"./Data/labelled_gestures_{domain_suffix}_experimental.csv"
        labelled_gestures.to_csv(output_filename, index=False)
        return labelled_gestures
    else:
        if verbose:
            print("Cross-validation clustering:")
        logo = LeaveOneGroupOut()
        groups = dataset["subject_id"]
        all_labeled = []
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(dataset, groups=groups)):
            if verbose:
                print(f"  Fold {fold_idx + 1}/10...")
            train_data = dataset.iloc[train_idx]
            test_data = dataset.iloc[test_idx]
            kmeans = clustering(number_of_clusters, train_data)
            labelled_train = label_conversion(train_data, kmeans)
            labelled_test = label_conversion(test_data, kmeans)
            all_labeled.append(pd.concat([labelled_train, labelled_test]))
        if verbose:
            print("Combining and saving labelled gestures...")
        labelled_data = pd.concat(all_labeled, ignore_index=True)

        if verbose:
            print('Saving labelled gestures...')
        labelled_gestures = label_gestures(labelled_data)
        output_filename = f"./Data/labelled_gestures_{domain_suffix}.csv"
        labelled_gestures.to_csv(output_filename, index=False)
        return labelled_gestures


def main():
    parser = argparse.ArgumentParser(description="Convert 3D gesture data to labelled sequences.")
    parser.add_argument("--domains", nargs="+", type=int, default=[1],
                        help="List of domain IDs to process (default: [1])")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output with progress messages.")
    parser.add_argument("--rigorous", action="store_true",
                        help="Use rigorous clustering: perform clustering inside cross-validation.")
    args = parser.parse_args()

    convert_3d_to_labels(args.domains, verbose=args.verbose, experimental=not args.rigorous)


if __name__ == "__main__":
    main()
