import numpy as np
import pandas as pd
from scipy import linalg
from pathlib import Path

# Read the data
DATA_DIR = Path.cwd() / "testfiles_" / "data"
csv_path = DATA_DIR / "problem5.csv"
data = pd.read_csv(csv_path, header=0)

X = data.values
n, p = X.shape

print("Q5")

# 5a
print("5a")

cov_matrix = np.zeros((p, p))

for i in range(p):
    for j in range(p):
        # Get valid pairs (no NaN in either column)
        valid_mask = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
        valid_pairs = X[valid_mask]

        if valid_pairs.shape[0] > 1:
            xi = valid_pairs[:, i]
            xj = valid_pairs[:, j]

            # Calculate covariance
            mean_i = np.mean(xi)
            mean_j = np.mean(xj)
            cov = np.sum((xi - mean_i) * (xj - mean_j)) / (len(xi) - 1)
            cov_matrix[i, j] = cov
        else:
            cov_matrix[i, j] = 0

print("\nPairwise Covariance Matrix:")
for i in range(p):
    for j in range(p):
        print(f"{cov_matrix[i, j]:10.6f}", end="  ")
    print()

# 5b
print("5b")

try:
    eigenvalues = linalg.eigvalsh(cov_matrix)
    print(f"\nEigenvalues: {eigenvalues}")

    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)

    tolerance = 1e-8

    if min_eig > tolerance:
        definiteness = "Positive Definite"
    elif min_eig >= -tolerance:
        definiteness = "Positive Semi-definite"
    else:
        definiteness = "Non Definite"

    print(f"\n>>> Matrix is: {definiteness}")

except Exception as e:
    print(f"Error computing eigenvalues: {e}")
    definiteness = "Unknown"

# 5c
print("5c")

def higham_nearestPSD(A, max_iter=100, tol=1e-8):
    """
    Higham's algorithm to find the nearest positive semi-definite matrix
    """
    n = A.shape[0]

    # Symmetrize
    A = (A + A.T) / 2

    # Initialize
    Y = A.copy()

    for iteration in range(max_iter):
        # Compute eigendecomposition
        eigvals, eigvecs = linalg.eigh(Y)

        # Check if already PSD
        if np.min(eigvals) >= tol:
            print(f"Converged in {iteration} iterations")
            return Y

        # Project eigenvalues to positive
        eigvals_pos = np.maximum(eigvals, tol)

        # Reconstruct matrix
        Y = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T

        # Symmetrize again
        Y = (Y + Y.T) / 2

    print(f"Reached max iterations ({max_iter})")
    return Y


if definiteness == "Non Definite":
    fixed_matrix = higham_nearestPSD(cov_matrix)

    print("\nFixed Covariance Matrix:")
    for i in range(p):
        for j in range(p):
            print(f"{fixed_matrix[i, j]:10.6f}", end="  ")
        print()

else:
    print("\nMatrix is already positive (semi-)definite. No fix needed.")
    fixed_matrix = cov_matrix

# 5d
print("5d")

# Use the fixed matrix for PCA
pca_matrix = fixed_matrix if definiteness == "Non Definite" else cov_matrix

eigenvalues_pca, eigenvectors_pca = linalg.eigh(pca_matrix)

# Sort by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues_pca)[::-1]
eigenvalues_sorted = eigenvalues_pca[sorted_indices]
eigenvectors_sorted = eigenvectors_pca[:, sorted_indices]

# Calculate variance explained
total_variance = np.sum(eigenvalues_sorted)
variance_explained = eigenvalues_sorted / total_variance
cumulative_variance = np.cumsum(variance_explained)


print(f"{'PC':<6} {'Eigenvalue':<15} {'Variance %':<15} {'Cumulative %':<15}")


for i in range(p):
    print(
        f"PC{i + 1:<4} {eigenvalues_sorted[i]:>14.6f} {variance_explained[i] * 100:>14.2f}% {cumulative_variance[i] * 100:>14.2f}%")

print("\nSummary:")
print(f"Total variance: {total_variance:.6f}")
print(f"PC1 explains {variance_explained[0] * 100:.2f}% of variance")
print(f"First 2 PCs explain {cumulative_variance[1] * 100:.2f}% of variance")
print(f"First 3 PCs explain {cumulative_variance[2] * 100:.2f}% of variance")
