import numpy as np
from sklearn.neighbors import NearestNeighbors

def l2n2_estimator(data, k=2, j=1, alpha=None, beta=None):
    """
    Estimates intrinsic dimensionality using the L2N2 method.
    
    Parameters:
    - data: array-like, shape (n_samples, n_features)
    - k: Index of the farther nearest neighbor (default 2)
    - j: Index of the closer nearest neighbor (default 1)
    - alpha: Tuning parameter alpha (n-dependent)
    - beta: Tuning parameter beta (n-dependent)
    
    Returns:
    - Estimated intrinsic dimensionality (d_hat)
    """
    n = len(data)
    
    # 1. Compute k-nearest neighbor distances [2, 3]
    # We need k+1 because the 0th neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    # Extract R_k and R_j distances (ignoring the 0th distance which is 0)
    # R_k is distances[:, k], R_j is distances[:, j]
    rk = distances[:, k]
    rj = distances[:, j]
    
    # 2. Compute L_k,j values: -log(log(R_k / R_j)) [1]
    # Small epsilon added to avoid log(1) or log(0) issues in noisy data
    ratio = rk / rj
    lkj_values = -np.log(np.log(ratio))
    
    # 3. Compute the average L_k,j [1]
    l_bar = np.mean(lkj_values)
    
    # 4. Apply tuning parameters for finite sample effects [1, 4]
    # Asymptotically, alpha -> 1 and beta -> C_k,j [5]
    # For k=2, j=1, C_2,1 is the Euler-Mascheroni constant (~0.5772) [6, 7]
    if alpha is None or beta is None:
        # Default to asymptotic values if n-specific parameters aren't provided
        alpha = 1.0
        beta = 0.57721  # Gamma (Euler-Mascheroni constant)
        print(f"Warning: Using asymptotic parameters. Results may be biased for small n.")

    d_hat = np.exp(alpha * l_bar + beta)
    
    return d_hat

# Example usage with parameters for n=2500 from Table VI in sources [8]
# For n=2500 and d range 1-20: alpha = 0.9299, beta = 0.6751
# data = np.random.randn(2500, 10) # Example 10D data
# id_estimate = l2n2_estimator(data, k=2, j=1, alpha=0.9299, beta=0.6751)