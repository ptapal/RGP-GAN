import numpy as np
from pyriemann.utils.mean import mean_riemann
from utils.spd_utils import ensure_spd

# has some data leakage when using the same data for training and testing instead of loso
class RiemannianParallelTransport:
    """
    Implements parallel transport on the SPD manifold to align multi-subject data
    while preserving task-related patterns. Inspiration fro paper on parallel transport: 
    "Parallel Transport on the Cone Manifold of SPD Matrices for Domain Adaptation".
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Initialize the parallel transport system.
        
        Args:
            epsilon: Numerical stability parameter
        """
        # ensure epsilon is a float otherwise i get an error for some reason
        self.epsilon = float(epsilon)
        self.global_reference = None
        self.subject_references = {} # store subject-specific reference points (Riemannian mean for each subject)
        self.transport_operators = {} # store the pre-computed transport operators for each subject seen during fitting
        self.is_fitted = False
        
    def _matrix_sqrt(self, C):
        """
        Compute the matrix square root (C^(1/2)) of an SPD matrix C using eigendecomposition.
        For an SPD matrix C = V * diag(lambda) * V.T, its square root is V * diag(sqrt(lambda)) * V.T. (linalg fundamental fact)
        This method should be more numerically stable than other methods like scipy.linalg.sqrtm for SPD matrices.

        Args:
            C: SPD matrix
            
        Returns:
            Matrix square root of C
        """
        # use eigendecomposition for numerical stability; 
        # compute eigenvalues (evals) and eigenvectors (evecs) of the symmetric matrix C.
        evals, evecs = np.linalg.eigh(C)
        
        # clip  any eigenvalue smaller than epsilon to protect against small negative eigenvalues
        # this should prevents issues with taking the square root of near-zero or potentially negative (due to numerical error) eigenvalues
        evals = np.maximum(evals, self.epsilon)
        
        # compute square root; reconstruct the matrix using eigenvectors and the square root of the (clipped) eigenvalues
        # np.diag creates a diagonal matrix from the eigenvalues' square roots
        sqrt_mat = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        
        return sqrt_mat
    
    def _matrix_sqrt_inv(self, C):
        """
        Compute the inverse of the matrix square root (C^(-1/2)) of an SPD matrix C.
        Similar to above _matrix_sqrt, but computes 1/sqrt(lambda) for the eigenvalues.
        C^(-1/2) = V * diag(1/sqrt(lambda)) * V.T
        
        Args:
            C: SPD matrix
            
        Returns:
            inverse of matrix square root of C
        """
        # use eigendecomposition for numerical stability
        evals, evecs = np.linalg.eigh(C)
        
        # protect against small eigenvalues
        evals = np.maximum(evals, self.epsilon)
        
        # compute inverse square root
        inv_sqrt_mat = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        
        return inv_sqrt_mat
    
    def fit(self, X, y, subjects=None, exclude_subject=None):
        """
        Compute the global reference point and subject-specific transport operators.

        Args:
            X: SPD matrices (n_samples, n_channels, n_channels)
            y: Class labels
            subjects: Subject identifiers
            exclude_subject: Subject to exclude from global reference (for LOSOCV)
            
        Returns:
            self: instance itself, allowing for method chaining
        """

        # exclude test subject data when computing global reference if specified
        if subjects is not None and exclude_subject is not None:
            # create mask for training subjects only
            train_mask = subjects != exclude_subject
            X_train = X[train_mask]

            # calculate global reference from training subjects only
            self.global_reference = mean_riemann(X_train)
        else:
            # if not doing LOSOCV or no subject info, compute the global reference using all provided data X.
            self.global_reference = mean_riemann(X)
        
        # ensure global reference is SPD
        self.global_reference = ensure_spd(self.global_reference, self.epsilon)
        
        # if subject information is provided, compute subject-specific references
        if subjects is not None:
            for subject_id in np.unique(subjects):
                subject_mask = subjects == subject_id
                subject_data = X[subject_mask]
                
                if len(subject_data) > 0:
                    # compute subject reference (mean across all classes)
                    subject_ref = mean_riemann(subject_data)
                    subject_ref = ensure_spd(subject_ref, self.epsilon)
                    self.subject_references[subject_id] = subject_ref
                    
                    # Compute S_k^(1/2)
                    sk_sqrt = self._matrix_sqrt(subject_ref)

                    # Compute S_k^(-1/2)
                    sk_inv_sqrt = self._matrix_sqrt_inv(subject_ref)

                    # Compute the inner term: X_0 = S_k^(-1/2) * G * S_k^(-1/2)
                    # S_k^(-1/2) is symmetric, so (S_k^(-1/2))^T = S_k^(-1/2)
                    # X_0 = sk_inv_sqrt @ self.global_reference @ sk_inv_sqrt  <-- ensures X_0 is symmetric
                    # X_0 needs to be SPD. ensure_spd might be good here?
                    X_0 = sk_inv_sqrt @ self.global_reference @ sk_inv_sqrt
                    X_0 = ensure_spd(X_0, self.epsilon) # Ensure the argument to sqrt is SPD

                    # Compute (X_0)^(1/2)
                    x0_sqrt = self._matrix_sqrt(X_0)

                    # Compute E = S_k^(1/2) * (X_0)^(1/2) * S_k^(-1/2)
                    transport_op = sk_sqrt @ x0_sqrt @ sk_inv_sqrt
                    transport_op = ensure_spd(transport_op, self.epsilon) # Ensure the operator E itself is SPD

                    # verify transport operator is well-conditioned !!!
                    cond = np.linalg.cond(transport_op)
                    if cond > 1e6:  # i just chose a random threshold for ill-conditioning
                        print(f"Warning: Transport operator for subject {subject_id} is ill-conditioned: {cond}")
                    
                    self.transport_operators[subject_id] = transport_op
        
        self.is_fitted = True
        return self
    
    def transform(self, X, subjects=None):
        """
        Apply the pre-computed parallel transport to align SPD matrices to the global reference frame.
        
        Args:
            X: SPD matrices to transform
            subjects: Subject identifiers
            
        Returns:
            X_aligned: Aligned SPD matrices
        """
        if not self.is_fitted:
            raise ValueError("ParallelTransport not fitted. Call fit() first!")
        
        X_aligned = np.zeros_like(X) # to store aligned matrices
        
        # if no subject info, align all to global reference
        if subjects is None:
            print("Warning: No subject information provided, no transformation done!")
            return X
        
        # track subject-specific transport operators for unseen subjects
        unseen_transport_ops = {}
        
        # sapply subject-specific transport operators
        for i in range(len(X)): # iterate through each input matrix and its corresponding subject ID
            subject_id = subjects[i]
            
            # get transport operator for this subject
            if subject_id in self.transport_operators:
                # use pre-computed transport operator
                transport_op = self.transport_operators[subject_id]
            else:
                # if subject not seen during fit, compute subject-specific mean first
                subject_mask = subjects == subject_id
                subject_samples = X[subject_mask]
                
                if len(subject_samples) > 1:
                    # compute mean reference for this subject
                    subject_ref = mean_riemann(subject_samples)
                    subject_ref = ensure_spd(subject_ref, float(self.epsilon))
                    
                    # compute transport operator
                    global_sqrt = self._matrix_sqrt(self.global_reference)
                    subject_isqrt = self._matrix_sqrt_inv(subject_ref)
                    transport_op = global_sqrt @ subject_isqrt
                else:
                    # fallback for single sample: treat as its own reference point S_k??
                    # this is not ideal but better than individual transport i think
                    print(f"Warning: Only one sample for subject {subject_id}, using as reference")
                    subject_sample = ensure_spd(X[i], float(self.epsilon))
                    global_sqrt = self._matrix_sqrt(self.global_reference)
                    subject_inv_sqrt = self._matrix_sqrt_inv(subject_sample)
                    transport_op = global_sqrt @ subject_inv_sqrt
                
                # keep for future use
                unseen_transport_ops[subject_id] = transport_op
            
            # apply transport!! apply the determined transport operator using the congruence transformation
            # X_aligned = T * X * T.T
            # op represents the linear mapping required to "move" data from the local geometric frame around the
            # subject's reference point (S_k) to the local geometric frame around the global reference point (G)
            X_aligned[i] = transport_op @ X[i] @ transport_op.T
            
            # ensure resulting matrix is SPD
            X_aligned[i] = ensure_spd(X_aligned[i], self.epsilon)
            
            # Explicitly enforce symmetry due to potential floating-point inaccuracies
            X_aligned[i] = (X_aligned[i] + X_aligned[i].T) / 2.0

            # verify symmetry
            if not np.allclose(X_aligned[i], X_aligned[i].T, rtol=1e-5, atol=1e-5):
                print(f"Warning: Symmetry broken after transport for sample {i}")
        
        return X_aligned
    
    
    def inverse_transform(self, X, subjects=None):
        """
        Apply the inverse parallel transport to map aligned matrices back towards their
        original subject-specific reference frames. T_inv = T^{-1}
        X_original = T_inv * X_aligned * T_inv.T

        Args:
            X: Aligned SPD matrices
            subjects: Subject identifiers
            
        Returns:
            X_original: SPD matrices in original reference frames
        """
        if not self.is_fitted:
            raise ValueError("ParallelTransport not fitted. Call fit() first.")
        
        X_original = np.zeros_like(X)
        
        # if no subject info, use identity transport
        if subjects is None:
            print("Warning: No subject info! Returning aligned SPD matrices as default!")
        
        # track unseen subject inverses
        unseen_inverse_ops = {}
        
        # apply inverse of subject-specific transport operators
        for i in range(len(X)):
            subject_id = subjects[i]
            
            # get transport operator for this subject
            if subject_id in self.transport_operators:
                # compute proper inverse (not transpose)
                transport_op = self.transport_operators[subject_id]
                inv_transport = np.linalg.inv(transport_op)
            else:
                # if subject not seen during fit, skip transport
                X_original[i] = X[i].copy()
                print("Warning: Subject not seen during fit.. Skipping inverse transport!")
                continue
            
            # apply inverse transport
            X_original[i] = inv_transport @ X[i] @ inv_transport.T
            
            # ensure resulting matrix is SPD
            X_original[i] = ensure_spd(X_original[i], self.epsilon)
        
        return X_original
    
    def verify_transport_reversibility(self, X, subjects=None, atol=1e-4):
        """
        Verify that transport followed by inverse transport recovers original matrices.
        
        Args:
            X: Original SPD matrices
            subjects: Subject identifiers
            atol: absolute tolerance 
            
        Returns:
            bool: True if transport is approximately reversible
        """
        if not self.is_fitted:
            raise ValueError("ParallelTransport not fitted. Call fit() first.")
        
        # transform and then inverse transform
        X_aligned = self.transform(X, subjects)
        X_recovered = self.inverse_transform(X_aligned, subjects)
        
        # check if original and recovered matrices are close
        for i in range(len(X)):
            if not np.allclose(X[i], X_recovered[i], atol=atol):
                relative_error = np.linalg.norm(X[i] - X_recovered[i]) / np.linalg.norm(X[i])
                print(f"Transport not reversible for sample {i}, relative error: {relative_error}")
                return False
        
        return True


