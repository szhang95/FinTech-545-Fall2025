import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t, norminvgauss
from scipy.optimize import minimize

plt.rcParams["figure.dpi"] = 150

# 0. Get Data
tickers = ["SPY", "TLT", "GLD"]
start_date = "2015-01-01"
end_date = "2025-12-01"

data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False  # keep "Adj Close"
)["Adj Close"]

data = data.dropna()

returns = np.log(data / data.shift(1)).dropna()
returns.columns = tickers

# Save datasets for reproducibility
data.to_csv("prices_SPY_TLT_GLD_2015_2025.csv")
returns.to_csv("logreturns_SPY_TLT_GLD_2015_2025.csv")

print("Head of daily log-returns:")
print(returns)

# 1. UNIVARIATE DESCRIPTIVE STATISTICS AND BASIC PLOTS
#    - Mean, Std, Skewness, Kurtosis (as in Week1 notes)
#    - Time series and histogram for SPY

summary = returns.describe().T
summary["skewness"] = returns.skew()
summary["kurtosis"] = returns.kurtosis()  # excess kurtosis
print("\nstatistics:")
print(summary)
summary.to_csv("summary_stats_returns.csv")

# SPY return time series (figure 1: showing volatility clustering and occasional extreme spikes.
plt.figure(figsize=(10, 4))
plt.plot(returns.index, returns["SPY"])
plt.title("Daily Log Returns - SPY")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.tight_layout()
plt.savefig("spy_returns_timeseries.png")

# SPY histogram + Normal fit (figure 2: showing the Normal model fits the center but underestimates tail thickness.)
plt.figure(figsize=(6, 4))
plt.hist(returns["SPY"], bins=60, density=True, alpha=0.6, label="Empirical")
xmin, xmax = returns["SPY"].min(), returns["SPY"].max()
x_grid = np.linspace(xmin, xmax, 500)
mu_spy_norm, sigma_spy_norm = norm.fit(returns["SPY"])
plt.plot(x_grid, norm.pdf(x_grid, mu_spy_norm, sigma_spy_norm),
         linewidth=2, label="Normal fit")
plt.title("SPY Returns Histogram + Normal Fit")
plt.xlabel("Log return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("spy_hist_normal.png")


# 2. ANALYSIS: FIT NORMAL, STUDENT-t, AND NIG DISTRIBUTIONS FOR SPY
#    - Maximum likelihood fitting for three distributions
#    - Log-likelihood and AIC comparison for model fit

spy_r = returns["SPY"].values

# Normal MLE
mu_norm, sigma_norm = norm.fit(spy_r)

# Student-t MLE
df_t_spy, loc_t_spy, scale_t_spy = t.fit(spy_r)

# NIG MLE
a_nig_spy, b_nig_spy, loc_nig_spy, scale_nig_spy = norminvgauss.fit(spy_r)

print("\nSPY Distribution Parameters")
print("Normal params: mu=%.6f, sigma=%.6f" % (mu_norm, sigma_norm))
print("t params: df=%.3f, loc=%.6f, scale=%.6f" %
      (df_t_spy, loc_t_spy, scale_t_spy))
print("NIG params: a=%.3f, b=%.3f, loc=%.6f, scale=%.6f" %
      (a_nig_spy, b_nig_spy, loc_nig_spy, scale_nig_spy))

# Log-likelihood and AIC comparison
def loglik_normal(x, mu, sigma):
    return np.sum(norm.logpdf(x, mu, sigma))

def loglik_t(x, df, loc, scale):
    return np.sum(t.logpdf(x, df, loc=loc, scale=scale))

def loglik_nig(x, a, b, loc, scale):
    return np.sum(norminvgauss.logpdf(x, a, b, loc=loc, scale=scale))

n_spy = len(spy_r)
k_norm = 2
k_t = 3
k_nig = 4

ll_norm = loglik_normal(spy_r, mu_norm, sigma_norm)
ll_t = loglik_t(spy_r, df_t_spy, loc_t_spy, scale_t_spy)
ll_nig = loglik_nig(spy_r, a_nig_spy, b_nig_spy, loc_nig_spy, scale_nig_spy)

aic_norm = 2 * k_norm - 2 * ll_norm
aic_t = 2 * k_t - 2 * ll_t
aic_nig = 2 * k_nig - 2 * ll_nig

print("\nLog-likelihoods (SPY):")
print(f"  Normal   : {ll_norm:.5f}")
print(f"  Student-t: {ll_t:.5f}")
print(f"  NIG      : {ll_nig:.5f}")

print("\nAIC (SPY):")
print(f"  Normal   : {aic_norm:.5f}")
print(f"  Student-t: {aic_t:.5f}")
print(f"  NIG      : {aic_nig:.5f}")


# Q-Q plot helper (figure 3,4,5)
def qq_plot(empirical, dist, params, title, filename):
    n = len(empirical)
    probs = (np.arange(1, n + 1) - 0.5) / n
    emp_sorted = np.sort(empirical)
    theo_quants = dist.ppf(probs, *params)

    plt.figure(figsize=(5, 5))
    plt.scatter(theo_quants, emp_sorted, s=5)
    min_val = min(emp_sorted.min(), theo_quants.min())
    max_val = max(emp_sorted.max(), theo_quants.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Empirical quantiles")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)

qq_plot(spy_r, norm, (mu_norm, sigma_norm),
        "Q-Q Plot: SPY vs Normal", "qq_SPY_normal.png")

qq_plot(spy_r, t, (df_t_spy, loc_t_spy, scale_t_spy),
        "Q-Q Plot: SPY vs Student-t", "qq_SPY_t.png")

qq_plot(spy_r, norminvgauss,
        (a_nig_spy, b_nig_spy, loc_nig_spy, scale_nig_spy),
        "Q-Q Plot: SPY vs NIG", "qq_SPY_nig.png")


# 3. ANALYSIS: 60/30/10 PORTFOLIO CONSTRUCTION AND FITTING
#    - Builds a static portfolio: 60% SPY, 30% TLT, 10% GLD
#    - Fits Normal, t, and NIG to portfolio returns

weights_603010 = np.array([0.6, 0.3, 0.1])
portfolio_ret = returns.dot(weights_603010)
portfolio_ret.name = "Portfolio_60_30_10"

print("\nPortfolio 60/30/10 Statistics")
print(portfolio_ret.describe())
print("Skewness:", portfolio_ret.skew())
print("Excess kurtosis:", portfolio_ret.kurtosis())

# Fit distributions for portfolio
r_p = portfolio_ret.values

mu_p_norm, sigma_p_norm = norm.fit(r_p)
df_p_t, loc_p_t, scale_p_t = t.fit(r_p)
a_p_nig, b_p_nig, loc_p_nig, scale_p_nig = norminvgauss.fit(r_p)

print("\nPortfolio Distribution Parameters")
print(
    "Normal:  mu = {:.5f}, sigma = {:.5f}"
    .format(mu_p_norm, sigma_p_norm)
)

print(
    "t:       df = {:.5f}, loc = {:.5f}, scale = {:.5f}"
    .format(df_p_t, loc_p_t, scale_p_t)
)


print(
    "NIG:     alpha = {:.5f}, beta = {:.5f}, delta = {:.5f}, mu = {:.5f}"
    .format(a_p_nig, b_p_nig, loc_p_nig, scale_p_nig)
)

params_norm_p = (mu_p_norm, sigma_p_norm)
params_t_p = (df_p_t, loc_p_t, scale_p_t)
params_nig_p = (a_p_nig, b_p_nig, loc_p_nig, scale_p_nig)

# 4. ANALYSIS: 1-DAY VAR & EXPECTED SHORTFALL (NORMAL / t / NIG)
#    - Uses VaR and ES definitions from Week4/Week5
#    - Normal ES uses the closed form formula in the notes
#    - t and NIG ES via simulation

def var_es_from_dist(dist, params, alpha):
    q = dist.ppf(1 - alpha, *params)
    var = -q
    n_sim = 100000
    sims = dist.rvs(*params, size=n_sim)
    tail_losses = -sims[sims <= q]
    es = tail_losses.mean()
    return var, es

rows_var_es = []
for alpha in [0.95, 0.99]:
    var_n, es_n = var_es_from_dist(norm, params_norm_p, alpha)
    var_t, es_t = var_es_from_dist(t, params_t_p, alpha)
    var_ng, es_ng = var_es_from_dist(norminvgauss, params_nig_p, alpha)

    print(f"\n{int(alpha*100)}% VaR / ES (1-day, portfolio)")
    print(f"Normal   VaR: {var_n:.5%}, ES: {es_n:.5%}")
    print(f"t        VaR: {var_t:.5%}, ES: {es_t:.5%}")
    print(f"NIG      VaR: {var_ng:.5%}, ES: {es_ng:.5%}")

    rows_var_es.append({
        "alpha": alpha,
        "Normal_VaR": var_n,
        "Normal_ES": es_n,
        "t_VaR": var_t,
        "t_ES": es_t,
        "NIG_VaR": var_ng,
        "NIG_ES": es_ng,
    })

var_es_df = pd.DataFrame(rows_var_es)
var_es_df.to_csv("portfolio_var_es_comparison.csv", index=False)
print("\nVaR/ES comparison table:")
print(var_es_df)

# 5. ANALYSIS: ROLLING 99% EXPECTED SHORTFALL (NORMAL VS NIG)
#    - Uses rolling 250-day window
#    - Normal ES: closed-form formula
#    - NIG ES: fitted NIG each window + simulation

window = 250  # ~1 trading year
alpha_es = 0.99

def nig_fit_es(x, alpha=0.99, n_sim=50000):
    a, b, loc, scale = norminvgauss.fit(x)
    sims = norminvgauss.rvs(a, b, loc=loc, scale=scale, size=n_sim)
    q = np.quantile(sims, 1 - alpha)
    es = -sims[sims <= q].mean()
    return es

def normal_es(mu, sigma, alpha=0.99):
    # ES formula for Normal from the notes
    z = stats.norm.ppf(1 - alpha)
    pdf = stats.norm.pdf(z)
    es = -(mu - sigma * pdf / (1 - alpha))
    return es

roll_es_normal = []
roll_es_nig = []
dates = []

r_arr = portfolio_ret.values

for i in range(window, len(r_arr)):
    window_data = r_arr[i-window:i]
    mu_w, sigma_w = norm.fit(window_data)
    es_n = normal_es(mu_w, sigma_w, alpha=alpha_es)
    es_ng = nig_fit_es(window_data, alpha=alpha_es, n_sim=30000)
    roll_es_normal.append(es_n)
    roll_es_nig.append(es_ng)
    dates.append(portfolio_ret.index[i])

roll_es_df = pd.DataFrame({
    "ES_Normal": roll_es_normal,
    "ES_NIG": roll_es_nig
}, index=pd.to_datetime(dates))

roll_es_df.to_csv("rolling_es_normal_vs_nig.csv")
print("\nrolling ES series:")
print(roll_es_df)


# # Figure 7: Rolling 99% ES comparison showing that NIG consistently assigns higher extreme-loss estimates than Normal, especially during crises.
plt.figure(figsize=(10, 4))
plt.plot(roll_es_df.index, roll_es_df["ES_Normal"], label="Normal ES (99%)")
plt.plot(roll_es_df.index, roll_es_df["ES_NIG"], label="NIG ES (99%)", linestyle="--")
plt.title("Rolling 99% Expected Shortfall - 60/30/10 Portfolio")
plt.xlabel("Date")
plt.ylabel("ES (loss, 1-day)")
plt.legend()
plt.tight_layout()
plt.savefig("rolling_es_normal_vs_nig.png")

# 6. ANALYSIS: MONTE CARLO SCENARIO SIMULATION (NORMAL / t / NIG)
#    - Simulates 1-day portfolio return distribution under each model
#    - Compares simulated VaR, ES, and tail loss probability

n_sim_mc = 100000

sim_norm = norm.rvs(*params_norm_p, size=n_sim_mc)
sim_t = t.rvs(*params_t_p, size=n_sim_mc)
sim_nig = norminvgauss.rvs(*params_nig_p, size=n_sim_mc)

def var_es_sample(sample, alpha=0.99):
    q = np.quantile(sample, 1 - alpha)
    var = -q
    es = -sample[sample <= q].mean()
    return var, es

for alpha in [0.95, 0.99]:
    print(f"\nSimulated VaR/ES at {int(alpha*100)}% (Portfolio)")
    for name, sims in [("Normal", sim_norm),
                       ("t", sim_t),
                       ("NIG", sim_nig)]:
        v, e = var_es_sample(sims, alpha=alpha)
        # Probability of a large loss, e.g. worse than -3 * sigma_p_norm
        prob_large_loss = (sims <= -3 * sigma_p_norm).mean()
        print(f"{name}: VaR={v:.4%}, ES={e:.4%}, "
              f"P(loss <= -3σ)={prob_large_loss:.4%}")

# Figure 6: Simulated 1-day portfolio return distributions under Normal, t, and NIG, highlighting that tail behavior differs even if the center looks similar.
plt.figure(figsize=(8, 4))
bins = 200
plt.hist(sim_norm, bins=bins, density=True, alpha=0.4, label="Normal")
plt.hist(sim_t, bins=bins, density=True, alpha=0.4, label="t")
plt.hist(sim_nig, bins=bins, density=True, alpha=0.4, label="NIG")
plt.xlim(-0.05, 0.05)
plt.title("Simulated 1-day Return Distribution (60/30/10 Portfolio)")
plt.xlabel("Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("simulated_distributions_portfolio.png")


# 7. ANALYSIS: RISK PARITY PORTFOLIO AND RISK CONTRIBUTIONS
#    - Uses covariance matrix (Week2 multivariate stats)
#    - Computes component risk contributions (Week8 attribution)
#    - Solves for risk parity weights via optimization

cov = returns.cov().values
n_assets = len(tickers)

def port_vol(w, cov):
    w = np.asarray(w)
    return np.sqrt(w.T @ cov @ w)

def risk_contribution(w, cov):
    """
    Component contribution to portfolio volatility:
    RC_i = w_i * (Σ w)_i / σ_p
    """
    w = np.asarray(w)
    sigma_p = port_vol(w, cov)
    marginal = cov @ w  # Σ w
    rc = w * marginal / sigma_p
    return rc, sigma_p

def risk_parity_objective(w, cov):
    rc, sigma_p = risk_contribution(w, cov)
    target = sigma_p / len(w)
    return ((rc - target) ** 2).sum()

x0 = np.ones(n_assets) / n_assets
bounds = [(0, 1)] * n_assets
cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

res = minimize(
    risk_parity_objective,
    x0,
    args=(cov,),
    bounds=bounds,
    constraints=cons,
    method="SLSQP",
    options={"disp": False}
)

w_rp = res.x
print("\nRisk Parity Weights")
for name, weight in zip(tickers, w_rp):
    print(f"{name}: {weight:.2%}")

rc_rp, sigma_rp = risk_contribution(w_rp, cov)
print("\nRisk Contributions (vol) for Risk Parity Portfolio:")
for name, rc_val in zip(tickers, rc_rp):
    print(f"{name}: {rc_val:.4%}")

rc_603010, sigma_603010 = risk_contribution(weights_603010, cov)
print("\nRisk Contributions for 60/30/10 Portfolio:")
for name, rc_val in zip(tickers, rc_603010):
    print(f"{name}: {rc_val:.4%}")
