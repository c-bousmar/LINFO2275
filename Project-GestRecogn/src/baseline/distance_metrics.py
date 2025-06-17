import numpy as np

def euclidean_distance(sequence1, sequence2):
    """
    Calculate the Euclidean distance between two sequences.
    
    @param sequence1: First sequence of points, with shape (n_points, n_dimensions)
    @param sequence2: Second sequence of points, with shape (m_points, n_dimensions)
    @return: The Euclidean distance between the two sequences
    """
    sequence1 = np.asarray(sequence1)
    sequence2 = np.asarray(sequence2)
    
    # Use the shorter sequence length
    min_length = min(len(sequence1), len(sequence2))
    
    # Truncate sequences to the same length
    seq1 = sequence1[:min_length]
    seq2 = sequence2[:min_length]
    
    # Calculate point-wise distances and sum them
    point_distances = np.sqrt(np.sum((seq1 - seq2) ** 2, axis=1))
    total_distance = np.sum(point_distances)
    
    return total_distance


def dtw_distance(seq1, seq2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    @param seq1: First sequence of points, with shape (n_points, n_dimensions)
    @param seq2: Second sequence of points, with shape (m_points, n_dimensions)
    @return: The DTW distance between the two sequences
    """
    n, m = len(seq1), len(seq2)

    # Initialize DTW matrix: dtw_matrix[i][j] = minimum cost to align seq1[:i] with seq2[:j]
    dtw_matrix = np.zeros((n + 1, m + 1))

    # Set initial values
    dtw_matrix[0, 0] = 0
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 1:] = np.inf

    # Fill the DTW matrix using dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            i_minus = i - 1
            j_minus = j - 1

            # Calculate cost as Euclidean distance between points
            cost = np.linalg.norm(seq1[i_minus] - seq2[j_minus])
            
            # Find minimum cost among three possible moves
            min_cost = min([
                dtw_matrix[i_minus, j],        # insertion (move in seq1)
                dtw_matrix[i, j_minus],        # deletion (move in seq2)
                dtw_matrix[i_minus, j_minus]   # match (move in both seq1 and seq2)
            ])

            # Update matrix with current cost plus minimum previous cost
            dtw_matrix[i, j] = cost + min_cost

    # Return the final DTW distance
    return dtw_matrix[n, m]


def edit_distance(seq1, seq2):
    """
    Compute the Edit Distance (Levenshtein distance) between two sequences.

    @param seq1: First sequence to compare
    @param seq2: Second sequence to compare
    @return: Minimum number of edit operations required to convert seq1 into seq2
    """
    # Ensure optimal computation by making seq1 the longer sequence
    if len(seq1) < len(seq2):
        return edit_distance(seq2, seq1)
    
    # Initialize previous row (represents empty seq1)
    previous = list(range(len(seq2) + 1))
    
    # Dynamic programming approach
    for i, c1 in enumerate(seq1):
        # Initialize current row
        current = [i + 1]
        
        for j, c2 in enumerate(seq2):
            # Calculate costs for different operations
            insertions = previous[j + 1] + 1      # Insert character from seq2
            deletions = current[j] + 1            # Delete character from seq1
            substitutions = previous[j] + (c1 != c2)  # 0 cost if characters match, 1 if not
            
            # Select minimum cost operation
            current.append(min(insertions, deletions, substitutions))
        
        # Update previous row for next iteration
        previous = current
    
    # Return final edit distance
    return previous[-1]


def lcs_length(seq1, seq2):
    """
    Compute the length of the Longest Common Subsequence between two sequences.

    @param seq1: First symbolic sequence
    @param seq2: Second symbolic sequence
    @return: Length of the longest common subsequence
    """
    m, n = len(seq1), len(seq2)
    
    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                # If characters match, increment the LCS
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                # Otherwise, take the maximum of excluding either character
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    
    # Return the length of the LCS
    return dp[m][n]


def lcs_distance(seq1, seq2):
    """
    Compute the normalized Longest Common Subsequence distance between two sequences.

    @param seq1: First symbolic sequence
    @param seq2: Second symbolic sequence
    @return: Normalized LCS distance between the two sequences (0=identical, 1=no common subsequence)
    """
    # Handle edge case of empty sequences
    if not seq1 and not seq2:
        return 0.0
    
    # Calculate normalized distance: 1 - (LCS length / max sequence length)
    return 1.0 - lcs_length(seq1, seq2) / max(len(seq1), len(seq2))