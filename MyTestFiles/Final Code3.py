import numpy as np
import pandas as pd
from scipy.optimize import minimize

possible_paths_insample = [
    "testfiles_/data/problem3_insample.csv",
    "MyTestFiles/testfiles_/data/problem3_insample.csv",
    "/Users/apple/Desktop/FinTech-545-Fall2025/MyTestFiles/testfiles_/data/problem3_insample.csv",
]

possible_paths_outsample = [
    "testfiles_/data/problem3_outsample.csv",
    "MyTestFiles/testfiles_/data/problem3_outsample.csv",
    "/Users/apple/Desktop/FinTech-545-Fall2025/MyTestFiles/testfiles_/data/problem3_outsample.csv",
]

data_insample = None
for path in possible_paths_insample:
    try:
        data_insample = pd.read_csv(path)
        print(f"Successfully loaded in-sample data from: {path}\n")
        break
    except FileNotFoundError:
        continue

if data_insample is None:
    raise FileNotFoundError("Could not find problem3_insample.csv")

data_outsample = None
for path in possible_paths_outsample:
    try:
        data_outsample = pd.read_csv(path)
        print(f"Successfully loaded out-of-sample data from: {path}\n")
        break
    except FileNotFoundError:
        continue

if data_outsample is None:
    raise FileNotFoundError("Could not find problem3_outsample.csv")

# sort by date
data_insample["Date"] = pd.to_datetime(data_insample["Date"])
data_outsample["Date"] = pd.to_datetime(data_outsample["Date"])

data_insample = data_insample.sort_values("Date")
data_outsample = data_outsample.sort_values("Date")

asset_cols = [c for c in data_insample.columns if c != "Date"]
n_assets = len(asset_cols)

returns_in = data_insample[asset_cols].values  # monthly in-sample returns
returns_out = data_outsample[asset_cols].values  # monthly out-of-sample returns


def ew_covariance(returns, lam):
    """
    Exponentially weighted covariance matrix.

    returns: T x N array of returns
    lam: decay parameter (e.g., 0.97)
    """
    T, N = returns.shape
    # weights proportional to lam^(T-1 - t) so that recent data gets higher weight
    exponents = np.arange(T - 1, -1, -1)
    w = lam ** exponents
    w = w / w.sum()

    # weighted mean
    mean = (returns * w[:, None]).sum(axis=0)

    # demeaned
    demeaned = returns - mean

    # cov = sum_t w_t * (r_t - mean)(r_t - mean)'
    cov = (demeaned.T * w) @ demeaned
    return cov


def neg_sharpe_ratio(w, mu, cov, rf):
    """Negative Sharpe ratio for optimization."""
    port_ret = w @ mu
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol <= 0:
        return 1e10
    sr = (port_ret - rf) / port_vol
    return -sr


def risk_contributions(w, cov):
    """
    Absolute risk contributions to portfolio volatility.

    Returns:
        rc: vector of absolute contributions (sum(rc) = sigma_p)
        sigma_p: portfolio volatility
    """
    sigma_p = np.sqrt(w @ cov @ w)
    if sigma_p <= 0:
        return np.zeros_like(w), 0.0
    mrc = cov @ w  # marginal risk contribution
    rc = w * mrc / sigma_p
    return rc, sigma_p


def risk_parity_objective(w, cov):
    """Squared error from equal risk contributions."""
    rc, sigma_p = risk_contributions(w, cov)
    if sigma_p <= 0:
        return 1e10
    target = sigma_p / len(w)
    return ((rc - target) ** 2).sum()


def risk_attribution(w, cov_annual):
    """
    Risk attribution using an annualized covariance matrix.

    Returns:
        rc_abs: absolute contributions to annual volatility
        rc_pct: percentage contributions
        sigma_p: portfolio annual volatility
    """
    rc_abs, sigma_p = risk_contributions(w, cov_annual)
    if sigma_p > 0:
        rc_pct = rc_abs / sigma_p
    else:
        rc_pct = np.zeros_like(rc_abs)
    return rc_abs, rc_pct, sigma_p


#part a
print("part a")

risk_free_rate = 0.04
# In-sample monthly mean returns
mu_month = returns_in.mean(axis=0)

# Annualize expected return: (1 + er)^12 - 1
mu_annual = (1.0 + mu_month) ** 12 - 1.0

lam = 0.97
cov_month_ew = ew_covariance(returns_in, lam)

ew_cov_month_df = pd.DataFrame(cov_month_ew, index=asset_cols, columns=asset_cols)
# Annualize covariance: ewCovar * 12
cov_annual_ew = cov_month_ew * 12.0
ew_cov_annual_df = pd.DataFrame(cov_annual_ew, index=asset_cols, columns=asset_cols)
print("\nExponentially weighted covariance matrix (annualized):")
print(ew_cov_annual_df.to_string(float_format=lambda x: f"{x: .6e}"))

# Max Sharpe Ratio portfolio (long-only)
w0 = np.ones(n_assets) / n_assets
constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
bounds = tuple((0.0, 1.0) for _ in range(n_assets))

res_msr = minimize(
    neg_sharpe_ratio,
    w0,
    args=(mu_annual, cov_annual_ew, risk_free_rate),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-9, "maxiter": 1000},
)

if not res_msr.success:
    print("Warning: Max SR optimization did not fully converge:", res_msr.message)

w_msr = res_msr.x
ret_msr_annual = w_msr @ mu_annual
vol_msr_annual = np.sqrt(w_msr @ cov_annual_ew @ w_msr)
sr_msr = (ret_msr_annual - risk_free_rate) / vol_msr_annual

print("\nMax Sharpe Ratio Portfolio:")
for name, weight in zip(asset_cols, w_msr):
    print(f"  {name}: weight = {weight:.4f}")
print(f"  Expected annual return: {ret_msr_annual:.4%}")
print(f"  Expected annual volatility: {vol_msr_annual:.4%}")
print(f"  Sharpe ratio (annual, rf={risk_free_rate:.2%}): {sr_msr:.4f}")

# Risk Parity portfolio (long-only)
res_rp = minimize(
    risk_parity_objective,
    w0,
    args=(cov_annual_ew,),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-9, "maxiter": 2000},
)

if not res_rp.success:
    print("Warning: Risk parity optimization did not fully converge:", res_rp.message)

w_rp = res_rp.x
ret_rp_annual = w_rp @ mu_annual
vol_rp_annual = np.sqrt(w_rp @ cov_annual_ew @ w_rp)
sr_rp = (ret_rp_annual - risk_free_rate) / vol_rp_annual
rc_rp_exante, sigma_rp_exante = risk_contributions(w_rp, cov_annual_ew)

print("\nRisk Parity Portfolio:")
for name, weight, rc_abs in zip(asset_cols, w_rp, rc_rp_exante):
    print(f"  {name}: weight = {weight:.4f}")
print(f"  Expected annual return: {ret_rp_annual:.4%}")
print(f"  Expected annual volatility: {vol_rp_annual:.4%}")
print(f"  Sharpe ratio (annual, rf={risk_free_rate:.2%}): {sr_rp:.4f}")

#part b
print("part b")

# Ex-post return attribution
mu_out_month = returns_out.mean(axis=0)  # each stock mean monthly return

# Portfolio monthly return series (no rebalancing)
port_ret_msr_month = returns_out @ w_msr
port_ret_rp_month = returns_out @ w_rp

# Portfolio annualized ex-post returns
er_msr_month = port_ret_msr_month.mean()
er_rp_month = port_ret_rp_month.mean()

er_msr_annual = (1.0 + er_msr_month) ** 12 - 1.0
er_rp_annual = (1.0 + er_rp_month) ** 12 - 1.0

# Per-asset monthly contribution to expected return: w_i * mu_i_out
contrib_msr_month = w_msr * mu_out_month
contrib_rp_month = w_rp * mu_out_month

# Pct of total monthly contribution
pct_contrib_msr = contrib_msr_month / contrib_msr_month.sum()
pct_contrib_rp = contrib_rp_month / contrib_rp_month.sum()

# Scale to annual
contrib_msr_annual = pct_contrib_msr * er_msr_annual
contrib_rp_annual = pct_contrib_rp * er_rp_annual

print("\nEx-post Return Attribution (annualized):")
print("Max Sharpe Ratio Portfolio:")
for name, abs_c, pct_c in zip(asset_cols, contrib_msr_annual, pct_contrib_msr):
    print(f"  {name}: abs contribution = {abs_c:.4%}, percent of total = {pct_c:.4%}")
print(f"  Total portfolio annual return (ex-post): {er_msr_annual:.4%}")

print("\nRisk Parity Portfolio:")
for name, abs_c, pct_c in zip(asset_cols, contrib_rp_annual, pct_contrib_rp):
    print(f"  {name}: abs contribution = {abs_c:.4%}, percent of total = {pct_c:.4%}")
print(f"  Total portfolio annual return (ex-post): {er_rp_annual:.4%}")

# Ex-post risk attribution
# Out-of-sample sample covariance (monthly) and annualize
cov_out_month = np.cov(returns_out, rowvar=False, ddof=1)
cov_out_annual = cov_out_month * 12.0

rc_msr_abs, rc_msr_pct, sigma_msr_expost = risk_attribution(w_msr, cov_out_annual)
rc_rp_abs, rc_rp_pct, sigma_rp_expost = risk_attribution(w_rp, cov_out_annual)

print("\nEx-post Risk Attribution (annual volatility):")
print("Max Sharpe Ratio Portfolio:")
for name, abs_rc, pct_rc in zip(asset_cols, rc_msr_abs, rc_msr_pct):
    print(f"  {name}: abs contribution = {abs_rc:.4%}, percent of total = {pct_rc:.4%}")
print(f"  Total portfolio volatility (ex-post, annual): {sigma_msr_expost:.4%}")

print("\nRisk Parity Portfolio:")
for name, abs_rc, pct_rc in zip(asset_cols, rc_rp_abs, rc_rp_pct):
    print(f"  {name}: abs contribution = {abs_rc:.4%}, percent of total = {pct_rc:.4%}")
print(f"  Total portfolio volatility (ex-post, annual): {sigma_rp_expost:.4%}")

# comparison
print("\nComparison: Ex-ante vs Ex-post (summary)")
print("Max Sharpe Ratio Portfolio:")
print(f"  Ex-ante annual return: {ret_msr_annual:.4%}, Ex-post: {er_msr_annual:.4%}")
print(f"  Ex-ante annual vol:    {vol_msr_annual:.4%}, Ex-post: {sigma_msr_expost:.4%}")

print("\nRisk Parity Portfolio:")
print(f"  Ex-ante annual return: {ret_rp_annual:.4%}, Ex-post: {er_rp_annual:.4%}")
print(f"  Ex-ante annual vol:    {vol_rp_annual:.4%}, Ex-post: {sigma_rp_expost:.4%}")
