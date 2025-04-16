import numpy as np

def euclidean_distance(seq1, seq2):
    """
    Calculate the Euclidean distance between two sequences of points.

    This function computes the Euclidean distance between two sequences `seq1` and `seq2`. 
    Each sequence is a list of points (or vectors), where each point is a vector in an N-dimensional space.
    
    Parameters
    ----------
    seq1 : list or np.ndarray
        The first sequence of points (e.g., a list of vectors or 2D numpy array).
    seq2 : list or np.ndarray
        The second sequence of points (e.g., a list of vectors or 2D numpy array).

    Returns
    -------
    float
        The Euclidean distance between `seq1` and `seq2`.

    Notes
    -----
    - The function assumes that `seq1` and `seq2` are sequences (lists or numpy arrays) of points/vectors.
    - Each point in the sequence must be represented as a numpy array or list (e.g., 2D arrays like [x, y, z]).
    """
    n = len(seq1)
    
    total_distance = 0.0
    for i in range(n):
        total_distance += np.linalg.norm(seq1[i] - seq2[i])

    return total_distance

def DTW(x, y):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    Dynamic Time Warping is a distance measure that compares two sequences by considering 
    the minimum cumulative distance between them. It is often used for time-series data 
    that may be misaligned or stretched in time.

    Parameters
    ----------
    x : np.ndarray
        A 2D numpy array representing the first sequence.
    y : np.ndarray
        A 2D numpy array representing the second sequence.

    Returns
    -------
    float
        The DTW distance between the two sequences.
    
    Notes
    -----
    - The function assumes that `x` and `y` are two 2D numpy arrays, where each row represents
      a point in a sequence (e.g., `(x1, y1, z1)` for the first sequence).
    - It uses Euclidean distance as the metric for calculating the local cost between points.
    """
    n, m = len(x), len(y)

    dtw_matrix = np.zeros((n + 1, m + 1))
    
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            
            i_minus = i - 1
            j_minus = j - 1
            
            cost = euclidean_distance(x[i_minus], y[j_minus])
            
            min_cost = min([dtw_matrix[i_minus, j],
                            dtw_matrix[i, j_minus],
                            dtw_matrix[i_minus, j_minus]])
            
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
    n, m = len(seq1), len(seq2)
    
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
        
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            
            i_minus = i - 1
            j_minus = j - 1
            
            if i_minus >= 0 and j_minus >= 0:
                cost = 0 if np.allclose(seq1[i_minus], seq2[j_minus], atol=1e-8) else 1
            else:
                cost = 1

            dp[i][j] = min(
                dp[i_minus][j] + 1,
                dp[i][j_minus] + 1,
                dp[i_minus][j_minus] + cost
            )
    
    return dp[n][m]