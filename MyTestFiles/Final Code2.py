import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# LOAD DATA
possible_paths = [
    'testfiles_/data/problem2.csv',
    'MyTestFiles/testfiles_/data/problem2.csv',
    '/Users/apple/Desktop/FinTech-545-Fall2025/MyTestFiles/testfiles_/data/problem2.csv'
]

data = None
for path in possible_paths:
    try:
        data = pd.read_csv(path)
        print(f"Successfully loaded data from: {path}\n")
        break
    except FileNotFoundError:
        continue

if data is None:
    raise FileNotFoundError("Could not find problem2.csv")

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# Calculate arithmetic returns (DO NOT remove mean)
data['Return'] = data['SPY'].pct_change()
returns = data['Return'].dropna()


# part a
print("part a")
# Fit normal
mu_normal = returns.mean()
sigma_normal = returns.std()

# Fit T
params_t = stats.t.fit(returns)
df_t, loc_t, scale_t = params_t

# log-likelihoods
ll_normal = np.sum(stats.norm.logpdf(returns, loc=mu_normal, scale=sigma_normal))
ll_t = np.sum(stats.t.logpdf(returns, df=df_t, loc=loc_t, scale=scale_t))

# AIC and BIC
n = len(returns)
k_normal = 2  # mu, sigma
k_t = 3  # df, loc, scale

aic_normal = 2 * k_normal - 2 * ll_normal
aic_t = 2 * k_t - 2 * ll_t
bic_normal = k_normal * np.log(n) - 2 * ll_normal
bic_t = k_t * np.log(n) - 2 * ll_t

print("\nNormal Distribution:")
print(f"  Mean: {mu_normal:.6f}")
print(f"  Std Dev: {sigma_normal:.6f}")
print(f"  Log-Likelihood: {ll_normal:.2f}")
print(f"  AIC: {aic_normal:.2f}")
print(f"  BIC: {bic_normal:.2f}")

print("\nT-Distribution:")
print(f"  Degrees of Freedom: {df_t:.2f}")
print(f"  Mean: {loc_t:.6f}")
print(f"  Scale: {scale_t:.6f}")
print(f"  Log-Likelihood: {ll_t:.2f}")
print(f"  AIC: {aic_t:.2f}")
print(f"  BIC: {bic_t:.2f}")

# part b
print("part b")

# Option parameters
S0 = data['SPY'].iloc[-1]  # Current stock price (most recent)
K_call = 665
K_put = 655
T_option = 10 / 255
r = 0.04
q = 0.0109
price_call = 7.05
price_put = 7.69

# Black-Scholes for Euro options w/th continuous dividends
def bs_call(S, K, T, r, q, sigma):
    if T <= 0:
        return np.maximum(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call


def bs_put(S, K, T, r, q, sigma):
    if T <= 0:
        return np.maximum(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    return put


# Implied volatility solver
def implied_vol(market_price, S, K, T, r, q, option_type='call'):
    def objective(sigma):
        if option_type == 'call':
            model_price = bs_call(S, K, T, r, q, sigma)
        else:
            model_price = bs_put(S, K, T, r, q, sigma)
        return (model_price - market_price) ** 2

    result = minimize(objective, x0=0.2, bounds=[(0.001, 5)], method='L-BFGS-B')
    return result.x[0]

iv_call = implied_vol(price_call, S0, K_call, T_option, r, q, 'call')
iv_put = implied_vol(price_put, S0, K_put, T_option, r, q, 'put')

print(f"Implied volatility (Call, K={K_call:.1f}): {iv_call * 100:.4f}%")
print(f"Implied volatility (Put,  K={K_put:.1f}): {iv_put * 100:.4f}%")

# part c
print("part c")

# Long 1 stock, Long 1 put, Short 1 call
port_value_current = S0 + price_put - price_call

print(f"\nCurrent Portfolio :")
print(f"  Long 1 share SPY: ${S0:.2f}")
print(f"  Long 1 put (K={K_put}): +${price_put:.2f}")
print(f"  Short 1 call (K={K_call}): -${price_call:.2f}")
print(f"  Total Portfolio Value: ${port_value_current:.2f}")

holding_days = 5
T_remaining = (10 - holding_days) / 255

# Monte Carlo simulation
np.random.seed(42)
n_simulations = 100000

daily_returns = stats.t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=(n_simulations, holding_days))

cumulative_returns = np.sum(daily_returns, axis=1)
S_future = S0 * np.exp(cumulative_returns)

#option price
call_future = np.array([bs_call(S, K_call, T_remaining, r, q, iv_call) for S in S_future])
put_future = np.array([bs_put(S, K_put, T_remaining, r, q, iv_put) for S in S_future])

# Portfolio value at end
port_value_future = S_future + put_future - call_future

# P&L
pnl = port_value_future - port_value_current

# VaR and ES at 5% level
# Loss = negative P&L (when portfolio loses value)
losses = -pnl
var_95 = np.percentile(losses, 95)  # 95th percentile of losses
es_95 = losses[losses >= var_95].mean()  # Expected loss beyond VaR

print(f"\nRisk Metrics (5% significance, absolute %):")
print(f"  VaR_5% (absolute %): {var_95 / port_value_current * 100:.2f}%")
print(f"  ES_5%  (absolute %): {es_95 / port_value_current * 100:.2f}%")

# part d
print("part d")
# maximize (mean_pnl - rf) / ES

target_value = 659.67
rf_rate = 0.0  # Risk-free rate
rf_5days = rf_rate * (holding_days / 255)  # Risk-free return for 5 days

current_holdings = np.array([1.0, 1.0, -1.0])  # [stock, put, call]

# Calculate current portfolio metrics using same simulations
def calc_portfolio_metrics(holdings):
    """
    Calculate portfolio metrics given holdings
    holdings = [n_stock, n_put, n_call]
    Returns: mean_pnl, es, pnl_array, portfolio_value
    """
    n_stock, n_put, n_call = holdings

    # Portfolio value today
    value_today = n_stock * S0 + n_put * price_put + n_call * price_call

    # Portfolio value +5 days
    value_future = n_stock * S_future + n_put * put_future + n_call * call_future

    # P&L
    pnl_array = value_future - value_today

    # Mean P&L
    mean_pnl = np.mean(pnl_array)

    # ES
    losses = -pnl_array
    var = np.percentile(losses, 95)
    es = np.mean(losses[losses >= var])

    return mean_pnl, es, pnl_array, value_today

# Current portfolio metrics
curr_mean, curr_es, curr_pnl, curr_value = calc_portfolio_metrics(current_holdings)
curr_ratio = (curr_mean - rf_5days) / curr_es

# Optimization
def objective_function(holdings):
    mean_pnl, es, _, _ = calc_portfolio_metrics(holdings)
    if es <= 1e-8:
        return 1e10
    ratio = (mean_pnl - rf_5days) / es
    return -ratio  # Maximize ratio = minimize negative ratio


def constraint_value(holdings):
    """Portfolio value must equal target_value"""
    n_stock, n_put, n_call = holdings
    value = n_stock * S0 + n_put * price_put + n_call * price_call
    return value - target_value


constraints = [
    {'type': 'eq', 'fun': constraint_value}
]

bounds = [(-2, 2), (-2, 2), (-2, 2)]

print("\nRunning optimization...")

# Try multiple initial guesses
initial_guesses = [
    current_holdings,
    np.array([1.0, 0.0, 0.0]),  # All stock
    np.array([0.5, 0.5, 0.0]),  # Stock + put
    np.array([0.0, 1.0, -1.0]),  # Put-call combo
]

best_result = None
best_ratio = -np.inf

for i, x0 in enumerate(initial_guesses):
    # Adjust initial guess to satisfy constraint
    scale = target_value / (x0[0] * S0 + x0[1] * price_put + x0[2] * price_call)
    x0_adj = x0 * scale

    result = minimize(
        objective_function,
        x0_adj,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )

    if result.success:
        test_ratio = -result.fun
        if test_ratio > best_ratio:
            best_ratio = test_ratio
            best_result = result

opt_holdings = best_result.x
opt_mean, opt_es, opt_pnl, opt_value = calc_portfolio_metrics(opt_holdings)
opt_ratio = (opt_mean - rf_5days) / opt_es

print(f"\nOptimal Portfolio:")
print(f"  Holdings: Stock={opt_holdings[0]:.4f}, Put={opt_holdings[1]:.4f}, Call={opt_holdings[2]:.4f}")
print(f"  Value: ${opt_value:.2f} (target: ${target_value:.2f})")
print(f"  Mean P&L: ${opt_mean:.4f}")
print(f"  ES (95%): ${opt_es:.2f}")
print(f"  Optimal Ratio: {opt_ratio:.6f}")


# part e
print("part e")

# Decide whether to use optimal or current portfolio
# Use optimal portfolio if optimization succeeded, otherwise use current
if ('best_result' in globals()) and (best_result is not None) and best_result.success:
    use_optimal = True
else:
    use_optimal = False

# Choose P&L series and mean for plotting
plot_pnl = opt_pnl if use_optimal else curr_pnl
plot_mean = opt_mean if use_optimal else curr_mean

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left plot: P&L distribution
axes[0].hist(plot_pnl, bins=100, alpha=0.7, color='steelblue',
             edgecolor='black', density=True)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2,
                label='Break-even', zorder=5)
axes[0].axvline(plot_mean, color='green', linestyle='--', linewidth=2,
                label=f'Mean: ${plot_mean:.2f}', zorder=5)
axes[0].set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
axes[0].set_title('Portfolio P&L Distribution (5-day holding period)',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right plot: Cumulative distribution
sorted_pnl = np.sort(plot_pnl)
cdf = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
axes[1].plot(sorted_pnl, cdf, linewidth=2, color='steelblue')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2,
                label='Break-even', zorder=5)
axes[1].axhline(0.05, color='orange', linestyle=':', linewidth=1.5,
                label='5th percentile', alpha=0.7)
axes[1].axhline(0.95, color='orange', linestyle=':', linewidth=1.5,
                label='95th percentile', alpha=0.7)
axes[1].set_xlabel('P&L ($)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Distribution Function',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

