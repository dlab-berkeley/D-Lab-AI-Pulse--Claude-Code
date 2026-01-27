# Helper functions
# Utility code used across scripts
# Some of this might not be used anymore

import pandas as pd
import numpy as np
from scipy import stats

def standardize(x):
    """Standardize to mean 0, sd 1"""
    return (x - x.mean()) / x.std()

def winsorize(x, limits=(0.01, 0.99)):
    """Winsorize at given percentiles"""
    lower = x.quantile(limits[0])
    upper = x.quantile(limits[1])
    return x.clip(lower, upper)

def remove_outliers_iqr(df, column, multiplier=1.5):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

def calculate_vif(X):
    """Calculate Variance Inflation Factors"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data

def weighted_mean(x, weights):
    """Calculate weighted mean"""
    return np.average(x, weights=weights)

def bootstrap_ci(data, func=np.mean, n_boot=1000, ci=95):
    """Bootstrap confidence interval"""
    boot_stats = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(func(sample))

    lower = np.percentile(boot_stats, (100-ci)/2)
    upper = np.percentile(boot_stats, 100 - (100-ci)/2)

    return lower, upper

# Old function - not sure if still used
def calc_summary_stats_old(df, cols):
    results = {}
    for c in cols:
        results[c] = {
            'mean': df[c].mean(),
            'sd': df[c].std()
        }
    return results

# From stack overflow - formats p-values nicely
def format_pvalue(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '+'
    else:
        return ''
