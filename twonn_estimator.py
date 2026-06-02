import numpy as np
from sklearn.neighbors import NearestNeighbors

def twonn_estimator(data, discard_fraction=0.1):
    """
    Estimates intrinsic dimensionality using the TWO-NN method.
    
    Parameters:
    - data: array-like, shape (n_samples, n_features)
    - discard_fraction: Fraction of points with the largest mu to discard (default 0.1)
    
    Returns:
    - Estimated intrinsic dimensionality (d_hat)
    """
    N = len(data)
    
    # 1. Compute pairwise distances and find the two shortest distances (r1, r2)
    # We need n_neighbors=3 because the 0th neighbor is the point itself [4, 6]
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # 2. Compute mu = r2 / r1 for each point [4]
    # Add a tiny epsilon to r1 to avoid division by zero
    mu = r2 / (r1 + 1e-12)
    
    # 3. Sort mu values in ascending order to compute empirical cumulative distribution [4]
    mu_sorted = np.sort(mu)
    
    # 4. Compute the empirical cumulate F_emp(mu_i) = i / N [4]
    # Note: Using (i-1)/N or similar to ensure F < 1 for the log(1-F) calculation
    i = np.arange(1, N + 1)
    f_emp = i / N
    
    # 5. Discard the top fraction of points with the highest mu values [5]
    # This makes the fit more stable against outliers and heavy tails
    n_keep = int(N * (1 - discard_fraction))
    mu_final = mu_sorted[:n_keep]
    f_final = f_emp[:n_keep]
    
    # 6. Fit the coordinates {(log(mu_i), -log(1 - F_emp(mu_i)))} [4]
    # The relationship is defined as: -log(1 - F(mu)) = d * log(mu) [7]
    x = np.log(mu_final)
    y = -np.log(1 - f_final)
    
    # Perform a linear fit through the origin (y = d * x)
    # The slope d is calculated as: sum(x*y) / sum(x*x)
    d_hat = np.sum(x * y) / np.sum(x**2)
    
    return d_hat

# Example usage:
# data = np.random.randn(2500, 10) # 10D Gaussian data
# id_estimate = twonn_estimator(data)
# print(f"Estimated ID: {id_estimate:.2f}")