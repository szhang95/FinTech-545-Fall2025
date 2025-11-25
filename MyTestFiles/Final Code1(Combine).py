import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math


def monte_carlo_option_price_from_returns(S0, returns, K, r, T,
                                          option_type="call",
                                          use_log_returns=False):
    """
    Price a 1-day European option from simulated 1-day returns.

    If use_log_returns=False: S_T = S0 * (1 + r_i)
    If use_log_returns=True:  S_T = S0 * exp(r_i)
    """
    returns = np.asarray(returns)

    if use_log_returns:
        ST = S0 * np.exp(returns)
    else:
        ST = S0 * (1.0 + returns)

    if option_type.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoff = np.exp(-r * T) * payoff
    price = discounted_payoff.mean()
    return price


# Method 1: Simple returns

def norm_cdf(x):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal PDF."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def black_scholes_call_price(S0, K, r, T, sigma):
    """Black-Scholes-Merton call price (no dividends)."""
    if sigma <= 0 or T <= 0:
        return max(0.0, math.exp(-r * T) * (S0 - K))

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return call_price


def black_scholes_call_and_vega(S0, K, r, T, sigma):
    """Return (call_price, vega) for Newton-Raphson implied vol."""
    if sigma <= 0 or T <= 0:
        sigma = 1e-8

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    vega = S0 * norm_pdf(d1) * math.sqrt(T)
    return call_price, vega


def implied_vol_call_bs(price, S0, K, r, T, initial_sigma=0.2,
                        tol=1e-6, max_iter=100):
    """
    Implied vol for a call using Newton-Raphson on Black-Scholes.
    Returns np.nan if it fails to converge.
    """
    intrinsic = max(S0 - K * math.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-8:
        # Very close to intrinsic value, vol ~ 0
        return 0.0

    sigma = initial_sigma
    for _ in range(max_iter):
        call_price, vega = black_scholes_call_and_vega(S0, K, r, T, sigma)
        diff = call_price - price

        if abs(diff) < tol:
            return max(sigma, 0.0)

        if vega < 1e-8:
            break

        sigma = sigma - diff / vega

        # Keep sigma in a reasonable range
        if sigma <= 0:
            sigma = 1e-6
        if sigma > 5.0:
            sigma = 5.0

    return np.nan


def run_method1_simple_returns(returns, S0=100.0, T=1/255.0, r=0.04,
                               strikes_part_a=None, strike_range=None):
    """
    Method 1:
    - Treat returns as simple returns: S_T = S0 * (1 + r)
    - Use custom Newton-Raphson for implied vol
    """
    if strikes_part_a is None:
        strikes_part_a = [99, 100, 101]
    if strike_range is None:
        strike_range = np.arange(95, 106, 1)

    print("=" * 70)
    print("METHOD 1: Simple Returns + Custom Implied Vol")
    print("=" * 70)
    print(f"S0 = {S0}, T = {T:.6f} years, r = {r:.2%}")
    print(f"Number of simulated returns: {len(returns)}")
    print("Assumption: r is a simple return (S_T = S0 * (1 + r))")

    # Part (a): prices for 99, 100, 101
    print("\nPart (a) - Option prices from simulated returns (Method 1)")
    print(f"{'Strike':<10}{'Call Price':<15}{'Put Price':<15}")
    for K in strikes_part_a:
        call_price = monte_carlo_option_price_from_returns(
            S0, returns, K, r, T, option_type="call", use_log_returns=False
        )
        put_price = monte_carlo_option_price_from_returns(
            S0, returns, K, r, T, option_type="put", use_log_returns=False
        )
        print(f"{K:<10}{call_price:<15.6f}{put_price:<15.6f}")

    # Part (b): implied vol smile
    call_prices = []
    implied_vols = []

    for K in strike_range:
        mc_call_price = monte_carlo_option_price_from_returns(
            S0, returns, K, r, T, option_type="call", use_log_returns=False
        )
        call_prices.append(mc_call_price)

        iv = implied_vol_call_bs(mc_call_price, S0, K, r, T, initial_sigma=0.2)
        implied_vols.append(iv)

    call_prices = np.array(call_prices)
    implied_vols = np.array(implied_vols)

    print("\nPart (b) - Implied volatility by strike (Method 1)")
    print(f"{'Strike':<10}{'Call Price':<15}{'Implied Vol':<15}")
    for K, price, iv in zip(strike_range, call_prices, implied_vols):
        print(f"{K:<10}{price:<15.6f}{iv:<15.6f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(strike_range, implied_vols, marker="o", linestyle="-")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility (sigma)")
    plt.title("Implied Volatility Smile (Method 1: Simple Returns)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Method 2: Log returns + Black-Scholes

def run_method2_log_returns_scipy(returns, S0=100.0, T=1/255.0, r=0.04,
                                  strikes_part_a=None, strike_range=None):
    """
    Method 2:
    - Treat returns as log returns: S_T = S0 * exp(r)
    - Use SciPy Black-Scholes + Brent root finder for implied vol
    """
    from scipy.stats import norm
    from scipy.optimize import brentq

    if strikes_part_a is None:
        strikes_part_a = [99, 100, 101]
    if strike_range is None:
        strike_range = np.arange(95, 106, 1)

    print("=" * 70)
    print("METHOD 2: Log Returns + SciPy Implied Vol")
    print("=" * 70)
    print(f"S0 = {S0}, T = {T:.6f} years, r = {r:.2%}")
    print(f"Number of simulated returns: {len(returns)}")
    print("Assumption: r is a log return (S_T = S0 * exp(r))")

    final_prices = S0 * np.exp(returns)
    discount_factor = np.exp(-r * T)

    # annualized
    actual_vol = np.std(returns) * np.sqrt(255)
    print(f"Historical volatility (from returns, annualized): {actual_vol:.4%}")

    def monte_carlo_option_price_from_final(final_prices, K, discount_factor):
        call_payoffs = np.maximum(final_prices - K, 0.0)
        put_payoffs = np.maximum(K - final_prices, 0.0)
        call_price = discount_factor * np.mean(call_payoffs)
        put_price = discount_factor * np.mean(put_payoffs)
        return call_price, put_price

    # Part (a):
    print("\nPart (a) - Option prices from simulated returns (Method 2)")
    print(f"{'Strike':<10}{'Call Price':<15}{'Put Price':<15}")
    for K in strikes_part_a:
        call_price, put_price = monte_carlo_option_price_from_final(
            final_prices, K, discount_factor
        )
        print(f"{K:<10}{call_price:<15.6f}{put_price:<15.6f}")

    # Black-Scholes
    def bs_call(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def implied_vol_bs_brent(market_price, S, K, T, r):
        def objective(sigma):
            return bs_call(S, K, T, r, sigma) - market_price

        try:
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except Exception:
            return np.nan

    # Part (b):
    print("\nPart (b) - Implied volatility by strike (Method 2)")
    print(f"{'Strike':<10}{'Call Price':<15}{'Implied Vol (%)':<15}")

    strikes_list = []
    implied_vols_pct = []

    for K in strike_range:
        call_price, put_price = monte_carlo_option_price_from_final(
            final_prices, K, discount_factor
        )
        iv = implied_vol_bs_brent(call_price, S0, K, T, r)
        if not np.isnan(iv):
            strikes_list.append(K)
            implied_vols_pct.append(iv * 100.0)
            print(f"{K:<10}{call_price:<15.6f}{iv * 100.0:<15.2f}")

    # Plot smile
    plt.figure(figsize=(8, 5))
    plt.plot(strikes_list, implied_vols_pct, marker="o", linestyle="-",
             label="Implied Volatility")
    plt.axhline(y=actual_vol * 100.0, color="r", linestyle="--",
                label=f"Historical Vol ~ {actual_vol * 100.0:.2f}%")
    plt.axvline(x=S0, color="g", linestyle=":", alpha=0.6, label="ATM Strike")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility (%)")
    plt.title("Implied Volatility Smile (Method 2: Log Returns + SciPy)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Main: load data and run Method 1 by default

if __name__ == "__main__":

    # Common parameters
    S0 = 100.0
    T = 1.0 / 255.0
    r = 0.04
    strikes_part_a = [99, 100, 101]
    strike_range = np.arange(95, 106, 1)

    # Load data
    possible_paths = [
        "testfiles_/data/problem1.csv",
        "MyTestFiles/testfiles_/data/problem1.csv",
    ]

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(
            os.path.join(script_dir, "testfiles_", "data", "problem1.csv")
        )
    except NameError:
        pass

    possible_paths.append(
        "/Users/apple/Desktop/FinTech-545-Fall2025/MyTestFiles/testfiles_/data/problem1.csv"
    )

    data = None
    for path in possible_paths:
        try:
            data = pd.read_csv(path)
            print(f"Successfully loaded data from: {path}")
            break
        except FileNotFoundError:
            continue

    if data is None:
        raise FileNotFoundError("Could not find problem1.csv in any expected location")

    returns = data["r"].values

    # Run Method 1 (simple returns) by default
    run_method1_simple_returns(
        returns,
        S0=S0,
        T=T,
        r=r,
        strikes_part_a=strikes_part_a,
        strike_range=strike_range,
    )

    # Optional: run Method 2 (log returns + SciPy)
    RUN_METHOD2 = True  # set to True if you want Method 2

    if RUN_METHOD2:
        run_method2_log_returns_scipy(
            returns,
            S0=S0,
            T=T,
            r=r,
            strikes_part_a=strikes_part_a,
            strike_range=strike_range,
        )
