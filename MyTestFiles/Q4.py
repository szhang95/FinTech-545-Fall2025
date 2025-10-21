import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path.cwd() / "testfiles_" / "data"
csv_path = DATA_DIR / "problem4.csv"
data = pd.read_csv(csv_path, header=0)
n, p = data.shape

print("=" * 60)
print("Q4")
print("=" * 60)



# ------------------------------------------------------------
print("Question 4a")

LAMBDA = 0.94
num = data.select_dtypes(include=[np.number]).dropna()
X = num.to_numpy(float)
n, d = X.shape

w = (1 - LAMBDA) * LAMBDA ** np.arange(n - 1, -1, -1)
w = w / w.sum()

mu = (w[:, None] * X).sum(axis=0)
Xc = X - mu
S = (w[:, None] * Xc).T @ Xc

std = np.sqrt(np.diag(S))
eps = 1e-18
std = np.where(std < eps, eps, std)
R = S / np.outer(std, std)

EW94 = pd.DataFrame(R, index=num.columns, columns=num.columns)
print(EW94)


# ------------------------------------------------------------
print("Question 4b")

LAMBDA = 0.97
w = (1 - LAMBDA) * LAMBDA ** np.arange(n - 1, -1, -1)
w = w / w.sum()

mu = (w[:, None] * X).sum(axis=0)
Xc = X - mu
S97 = (w[:, None] * Xc).T @ Xc

ew_cov97 = pd.DataFrame(S97, index=num.columns, columns=num.columns)
print(ew_cov97)

# ------------------------------------------------------------
print("Question 4c")

std_devs2 = np.sqrt(np.diag(ew_cov97))

# outer product 3x3
outer_std = np.outer(std_devs2, std_devs2)

# covariance
cov_matrix_combined = EW94.values * outer_std
cov_matrix_combined = pd.DataFrame(cov_matrix_combined,
                                   index=num.columns,
                                   columns=num.columns)

print("#Covariance between EW Variance (lambda=0.97) and EW Correlation (lambda=0.94)")
print(cov_matrix_combined.round(6))
