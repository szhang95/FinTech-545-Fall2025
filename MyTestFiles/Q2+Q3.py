import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis
from pathlib import Path
import matplotlib.pyplot as plt

from library import *

#Q2

def problem_2():
    """
    Problem 2: Calculate Mean, Variance, Skewness and Kurtosis
    """
    print("Q2")

    data_path = Path('testfiles_/data/problem2.csv')
    df = pd.read_csv(data_path)
    X = df.iloc[:, 0].dropna().values


    # ========================================================================
    print("2a")

    mean = np.mean(X)
    variance = np.var(X, ddof=1)
    std = np.sqrt(variance)
    skewness = skew(X, bias=False)
    kurt = kurtosis(X, bias=False, fisher=True)

    print(f"  Mean:       {mean:.10f}")
    print(f"  Variance:   {variance:.10f}")
    print(f"  Skewness:   {skewness:.10f}")
    print(f"  Kurtosis:   {kurt:.10f}")




    # ========================================================================
    print("2c")

    print("\n1. Normal Distribution:")
    params_normal = fit_normal_distribution(X)
    mu_norm = params_normal['mean']
    sigma_norm = params_normal['std']
    print(f"   μ (mean) = {mu_norm:.10f}")
    print(f"   σ (std)  = {sigma_norm:.10f}")

    print("\n2. t Distribution:")
    params_t = fit_t_distribution(X)
    nu_t = params_t['df']
    mu_t = params_t['loc']
    sigma_t = params_t['scale']
    print(f"   ν (df)    = {nu_t:.10f}")
    print(f"   μ (loc)   = {mu_t:.10f}")
    print(f"   σ (scale) = {sigma_t:.10f}")

    print("\nLog-Likelihood Comparison:")
    ll_normal = np.sum(norm.logpdf(X, loc=mu_norm, scale=sigma_norm))
    ll_t = np.sum(t.logpdf(X, df=nu_t, loc=mu_t, scale=sigma_t))
    print(f"  Normal distribution: LL = {ll_normal:.4f}")
    print(f"  t distribution:      LL = {ll_t:.4f}")

    if ll_t < ll_normal:
        print(f"   → Normal distribution (LL is larger)")
        better_model = "normal distribution"
    else:
        print(f"   → t distribution (LL is larger)")
        better_model = "t distribution"

    k_normal = 2
    k_t = 3
    aic_normal = 2 * k_normal - 2 * ll_normal
    aic_t = 2 * k_t - 2 * ll_t

    print(f"\nAIC Comparison (less->better):")
    print(f"  Normal Distribution: AIC = {aic_normal:.4f}")
    print(f"  t Distribution:      AIC = {aic_t:.4f}")

    if aic_t < aic_normal:
        print(f"   → t distribution (AIC is less)")
        better_model = "t distribution"
    else:
        print(f"   → Normal distribution (AIC is less)")
        better_model = "normal distribution"

    # FIX: Return results for problem_3()
    return {
        'X': X,
        'params_normal': params_normal,
        'params_t': params_t
    }



# Q3

def problem_3(results_p2):
    """
    Problem 3: Calculate VaR and ES using fitted models
    """
    print("Q3")

    X = results_p2['X']
    params_normal = results_p2['params_normal']
    params_t = results_p2['params_t']
    alpha = 0.05


    print("3a")

    # Normal distribution
    mu_norm = params_normal['mean']
    sigma_norm = params_normal['std']
    z_alpha = norm.ppf(alpha)
    var_norm_level = mu_norm + sigma_norm * z_alpha
    var_norm_from_zero = abs(var_norm_level)

    print("Normal Distribution:")
    print(f"VaR Absolute: {var_norm_from_zero:.17f}\n")

    # t distribution
    nu_t = params_t['df']
    mu_t = params_t['loc']
    sigma_t = params_t['scale']
    t_alpha = t.ppf(alpha, nu_t)
    var_t_level = mu_t + sigma_t * t_alpha
    var_t_from_zero = abs(var_t_level)

    print("t-distribution:")
    print(f"VaR Absolute: {var_t_from_zero:.17f}\n")

    print("3b")

    # Normal distribution ES
    phi_z = norm.pdf(z_alpha)
    es_norm_level = mu_norm - sigma_norm * phi_z / alpha
    es_norm_from_zero = abs(es_norm_level)

    print("Normal Distribution:")
    print(f"ES Absolute: {es_norm_from_zero:.15f}\n")

    # t distribution ES
    from scipy.integrate import quad

    def integrand(x):
        return x * t.pdf(x, nu_t, mu_t, sigma_t)

    lower_bound = t.ppf(1e-10, nu_t, mu_t, sigma_t)
    integral, _ = quad(integrand, lower_bound, var_t_level, limit=200)
    es_t_level = integral / alpha
    es_t_from_zero = abs(es_t_level)

    print("t-distribution:")
    print(f"ES Absolute: {es_t_from_zero:.15f}\n")


def main():
    results_p2 = problem_2()
    results_p3 = problem_3(results_p2)


if __name__ == "__main__":
    main()
