import numpy as np

def euclidean_distance(sequence1, sequence2):
    """
    Calculate the Euclidean distance between two sequences.
    
    For sequences of different lengths, this function computes the sum of 
    Euclidean distances between corresponding points up to the length of
    the shorter sequence. This provides a way to compare sequences of
    different lengths while still capturing meaningful distance information.
    
    Parameters:
    -----------
    sequence1 : array-like
        First sequence of points, with shape (n_points, n_dimensions)
    sequence2 : array-like
        Second sequence of points, with shape (m_points, n_dimensions)
    
    Returns:
    --------
    float
        The Euclidean distance between the two sequences
    """
    sequence1 = np.asarray(sequence1)
    sequence2 = np.asarray(sequence2)
    
    min_length = min(len(sequence1), len(sequence2))
    
    seq1 = sequence1[:min_length]
    seq2 = sequence2[:min_length]
    
    point_distances = np.sqrt(np.sum((seq1 - seq2) ** 2, axis=1))
    total_distance = np.sum(point_distances)
    
    return total_distance

def dtw_distance(seq1, seq2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    Dynamic Time Warping is a distance measure that compares two sequences by considering 
    the minimum cumulative distance between them. It is often used for time-series data 
    that may be misaligned or stretched in time.

    Parameters
    ----------
    seq1 : np.ndarray
        A 2D numpy array representing the first sequence.
    seq2 : np.ndarray
        A 2D numpy array representing the second sequence.

    Returns
    -------
    float
        The DTW distance between the two sequences.
    
    Notes
    -----
    - The function assumes that `seq1` and `seq2` are two 2D numpy arrays, where each row represents
      a point in a sequence (e.g., `(x1, y1, z1)` for the first sequence).
    - It uses Euclidean distance as the metric for calculating the local cost between points.
    """
    n, m = len(seq1), len(seq2)

    # dtw_matrix[i][j] = minimum cost to align x[:i] with y[:j]
    dtw_matrix = np.zeros((n + 1, m + 1))

    dtw_matrix[0, 0] = 0
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf

    # Loop over each cell of the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            i_minus = i - 1
            j_minus = j - 1

            # Euclidean distance between the two points
            cost = np.linalg.norm(seq1[i_minus] - seq2[j_minus])
            
            min_cost = min([dtw_matrix[i_minus, j],         # insertion (move inside seq1)
                            dtw_matrix[i, j_minus],         # deletion (move inside seq2)
                            dtw_matrix[i_minus, j_minus]])  # match (move in both seq1 and seq2)

            dtw_matrix[i, j] = cost + min_cost

    return dtw_matrix[n, m]


def edit_distance(seq1, seq2):
    """
    Compute the Edit Distance (Levenshtein distance) between two sequences.

    The Edit Distance (Levenshtein distance) measures how many insertions, deletions, or substitutions
    are required to transform one sequence into another. It is widely used for text comparison or alignment.

    Parameters
    ----------
    seq1 : list or np.ndarray
        The first sequence to compare.
    seq2 : list or np.ndarray
        The second sequence to compare.

    Returns
    -------
    int
        The minimum number of edit operations (insertions, deletions, substitutions) required to convert `seq1` into `seq2`.

    Notes
    -----
    - This function assumes that the sequences `seq1` and `seq2` are one-dimensional arrays or lists, 
      where each element in the sequence can be compared for equality (e.g., strings or numerical data).
    - The function uses dynamic programming to compute the minimum number of edit operations.
    - A cost of 1 is assigned for substitutions when two elements are not equal, and 0 if they are equal.
    """
    if len(seq1) < len(seq2):
        return edit_distance(seq2, seq1)
    previous = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        current = [i + 1]
        for j, c2 in enumerate(seq2):
            insertions = previous[j + 1] + 1
            deletions = current[j] + 1
            substitutions = previous[j] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def lcs_length(seq1, seq2):
    """
    Computes the length of the Longest Common Subsequence (LCS) between two sequences (Theodoridis et al., 2009).

    This function uses dynamic programming to compute the length of the LCS between two
    symbol sequences (here resulting from vector quantization of 3D gestures).

    Args:
        seq1 (list of str): The first symbolic sequence.
        seq2 (list of str): The second symbolic sequence.

    Returns:
        int: The length of the longest common subsequence between the two sequences.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def lcs_distance(seq1, seq2):
    """
    Computes the normalized Longest Common Subsequence (LCS) distance between two sequences.

    The distance is defined as:
        1 - LCS(seq1, seq2) / max(len(seq1), len(seq2))
    It ranges from 0 (identical) to 1 (no common subsequence).

    Args:
        seq1 (list of str): The first symbolic sequence.
        seq2 (list of str): The second symbolic sequence.

    Returns:
        float: The normalized LCS distance between the two sequences.
    """
    if not seq1 and not seq2:
        return 0.0
    return 1.0 - lcs_length(seq1, seq2) / max(len(seq1), len(seq2))
