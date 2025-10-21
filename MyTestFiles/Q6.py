import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import cholesky
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Read the data
DATA_DIR = Path.cwd() / "testfiles_" / "data"
data = pd.read_csv(DATA_DIR / "problem6.csv")
prices = data[['x1', 'x2', 'x3']].values
n_obs, n_stocks = prices.shape

print("Q6")

# Calculate arithmetic returns
returns = np.diff(prices, axis=0) / prices[:-1]
n_returns = returns.shape[0]

# Problem 6a: De-mean returns and fit Student T model
print("6a")
# De-mean the returns
returns_mean = np.mean(returns, axis=0)
returns_demeaned = returns - returns_mean

"""print("\nOriginal mean returns:")
for i in range(n_stocks):
    print(f"Stock {i + 1}: {returns_mean[i]:.6f}")

print("\nDe-meaned returns mean:")
for i in range(n_stocks):
    print(f"Stock {i + 1}: {np.mean(returns_demeaned[:, i]):.10f}")"""


# Fit Student T distribution to each stock
def fit_student_t(data):
    """Fit Student T distribution using MLE"""
    # Initial guess
    df_init = 10
    loc_init = 0
    scale_init = np.std(data)

    # Fit using scipy
    params = stats.t.fit(data, floc=0)  # fix location at 0 (already demeaned)
    df, loc, scale = params

    return df, loc, scale


t_params = []
print("\nStudent T Distribution Parameters:")
print(f"{'Stock':<10} {'Degrees of Freedom':<25} {'Location':<15} {'Scale':<15}")

for i in range(n_stocks):
    df, loc, scale = fit_student_t(returns_demeaned[:, i])
    t_params.append((df, loc, scale))
    print(f"Stock {i + 1:<4} {df:>20.4f} {loc:>14.6f} {scale:>14.6f}")

# Problem 6b: Gaussian Copula - correlation matrix
print("6b")

# Transform to uniform using empirical CDF (rank transformation)
def empirical_cdf(data):
    """Convert data to uniform [0,1] using empirical CDF"""
    n = len(data)
    ranks = stats.rankdata(data)
    return ranks / (n + 1)

# For Gaussian copula, we need the correlation of the normal quantiles
uniform_data = np.zeros_like(returns_demeaned)
normal_data = np.zeros_like(returns_demeaned)

for i in range(n_stocks):
    # Transform to uniform using fitted t-distribution
    uniform_data[:, i] = stats.t.cdf(returns_demeaned[:, i],
                                     df=t_params[i][0],
                                     loc=t_params[i][1],
                                     scale=t_params[i][2])
    # Transform to standard normal
    normal_data[:, i] = stats.norm.ppf(np.clip(uniform_data[:, i], 1e-10, 1 - 1e-10))

# Calculate correlation matrix from normal data
correlation_matrix = np.corrcoef(normal_data.T)

print("\nCorrelation Matrix (for Gaussian Copula):")
for i in range(n_stocks):
    for j in range(n_stocks):
        print(f"{correlation_matrix[i, j]:8.4f}", end="  ")
    print()

# 6c+6d
n_simulations = 10000
alpha = 0.05
shares = 100
current_prices = prices[-1, :]
current_portfolio_value = np.sum(shares * current_prices)

# Generate correlated normal samples using Cholesky decomposition
L = cholesky(correlation_matrix, lower=True)

# Storage for simulated returns
simulated_returns = np.zeros((n_simulations, n_stocks))
simulated_pnl_individual = np.zeros((n_simulations, n_stocks))
simulated_pnl_portfolio = np.zeros(n_simulations)

np.random.seed(42)  # For reproducibility

for sim in range(n_simulations):
    # Generate independent standard normal
    z = np.random.standard_normal(n_stocks)

    # Apply correlation using Cholesky
    correlated_z = L @ z

    # Transform to uniform
    u = stats.norm.cdf(correlated_z)

    # Transform to Student T using inverse CDF
    for i in range(n_stocks):
        simulated_returns[sim, i] = stats.t.ppf(u[i],
                                                df=t_params[i][0],
                                                loc=t_params[i][1],
                                                scale=t_params[i][2])

    # Calculate P&L for each stock
    simulated_pnl_individual[sim, :] = shares * current_prices * simulated_returns[sim, :]

    # Calculate total portfolio P&L
    simulated_pnl_portfolio[sim] = np.sum(simulated_pnl_individual[sim, :])

# 6c
print("6c")

print(f"{'Stock':<10} {'VaR ($)':<15} {'ES ($)':<15}")


individual_vars = []
individual_es = []

for i in range(n_stocks):
    # Sort P&L (losses are negative)
    sorted_pnl = np.sort(simulated_pnl_individual[:, i])

    # VaR is the negative of the alpha-quantile
    var_index = int(np.floor(alpha * n_simulations))
    var = -sorted_pnl[var_index]

    # ES is the average of losses beyond VaR
    es = -np.mean(sorted_pnl[:var_index + 1])

    individual_vars.append(var)
    individual_es.append(es)

    print(f"Stock {i + 1:<4} ${var:>13.2f} ${es:>13.2f}")

# 6d
print("6d")

# Sort portfolio P&L
sorted_portfolio_pnl = np.sort(simulated_pnl_portfolio)

# Portfolio VaR
var_index = int(np.floor(alpha * n_simulations))
portfolio_var = -sorted_portfolio_pnl[var_index]

# Portfolio ES
portfolio_es = -np.mean(sorted_portfolio_pnl[:var_index + 1])

print(f"\nPortfolio VaR: ${portfolio_var:.2f}")
print(f"Portfolio ES:  ${portfolio_es:.2f}")
