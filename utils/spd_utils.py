import numpy as np
import torch
from pyriemann.utils.distance import distance_riemann, pairwise_distance


def ensure_spd(matrix, epsilon=1e-6):
    """Forces a matrix to be SPD by fixing negative eigenvalues"""
    # convert epsilon to float if it's not already (just a safety check)
    epsilon = float(epsilon)

    # get smallest eigenvalue
    min_eig = np.linalg.eigvalsh(matrix).min()

    # if smallest eigenvalue is too small, add to diagonal
    if min_eig < epsilon:
        matrix += (epsilon - min_eig) * np.eye(matrix.shape[0])
        #print('Corrected SPD')
    return matrix

def safe_distance_riemann(A, B, epsilon=1e-6):
    """Safely compute riemannian distance between matrices"""
    try:
        # ensure both matrices are SPD
        A = ensure_spd(A, epsilon)
        B = ensure_spd(B, epsilon)
        return distance_riemann(A, B)
    except:
        # return large penalty if computation fails
        return 100.0

def verify_spd_properties(matrices, verbose=True):
    """
    Verify SPD properties for a batch of matrices.
    Check for symmetry and positive definiteness.
    
    Args:
        matrices: Array of matrices to check
        verbose: Whether to print detailed information
        
    Returns:
        tuple: (n_not_symmetric, n_not_positive_definite)
    """
    n_matrices = matrices.shape[0]
    if verbose:
        print(f"Checking {n_matrices} matrices for SPD properties...")
    
    n_not_symmetric = 0
    n_not_positive_definite = 0
    
    for matrix in matrices:
        # symmetry check
        symm_error = np.abs(matrix - matrix.T).max()
        if symm_error > 1e-10:
            n_not_symmetric += 1
        
        # eigenvalue check
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals <= 0):
            n_not_positive_definite += 1
    
    if verbose:
        print(f"Non-symmetric matrices: {n_not_symmetric} ({n_not_symmetric/n_matrices*100:.2f}%)")
        print(f"Non-positive definite matrices: {n_not_positive_definite} ({n_not_positive_definite/n_matrices*100:.2f}%)")
    
    return n_not_symmetric, n_not_positive_definite

def filter_spd_matrices(matrices: np.ndarray, labels: np.ndarray, verbose: bool=True):
    """
    Filters out any non-SPD matrices from a dataset.

    Args:
        matrices: The array of matrices to check
        labels: The corresponding labels
        verbose: Whether to print information about removed matrices

    Returns:
        tuple: A tuple containing the filtered matrices and filtered labels
    """
    valid_indices = []
    for i, matrix in enumerate(matrices):
        # vheck for symmetry and positive definiteness
        is_symmetric = np.allclose(matrix, matrix.T)
        try:
            # add a small epsilon for stability in eigenvalue check
            is_positive_definite = np.linalg.eigvalsh(matrix).min() > 1e-10
        except np.linalg.LinAlgError:
            is_positive_definite = False # mmatrix is singular or has other issues

        if is_symmetric and is_positive_definite:
            valid_indices.append(i)

    num_removed = len(matrices) - len(valid_indices)
    if num_removed > 0 and verbose:
        print(f"Removed {num_removed} non-SPD matr(ix/ices) from the dataset before classification!")

    return matrices[valid_indices], labels[valid_indices]