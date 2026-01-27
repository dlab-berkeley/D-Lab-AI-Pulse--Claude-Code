# Robustness checks
# Added after R&R from journal
# September 2024

import pandas as pd
import numpy as np
import statsmodels.api as sm

DATA_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv'
OUTPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/outputs'

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def robustness_age_restriction():
    """Reviewer 1: Exclude very young respondents"""
    df = load_data()
    df = df[df['age'] >= 25]

    X = df[['age', 'female', 'education_years', 'income', 'employed', 'trust_index']]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    with open(f'{OUTPUT_PATH}/robustness_age25.txt', 'w') as f:
        f.write("Robustness: Excluding respondents under 25\n\n")
        f.write(model.summary().as_text())

    return model

def robustness_income_winsorize():
    """Reviewer 1: Winsorize income at 1/99 percentiles"""
    df = load_data()

    p1 = df['income'].quantile(0.01)
    p99 = df['income'].quantile(0.99)
    df['income'] = df['income'].clip(p1, p99)

    X = df[['age', 'female', 'education_years', 'income', 'employed', 'trust_index']]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    with open(f'{OUTPUT_PATH}/robustness_winsorize.txt', 'w') as f:
        f.write("Robustness: Income winsorized at 1st/99th percentiles\n\n")
        f.write(model.summary().as_text())

    return model

def robustness_ordered_logit():
    """Reviewer 2: Use ordered logit instead of OLS"""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    df = load_data()

    # Satisfaction is 1-10, treat as ordinal
    X = df[['age', 'female', 'education_years', 'income', 'employed', 'trust_index']]
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = OrderedModel(y, X, distr='logit').fit(method='bfgs')

    with open(f'{OUTPUT_PATH}/robustness_ologit.txt', 'w') as f:
        f.write("Robustness: Ordered Logit Model\n\n")
        f.write(model.summary().as_text())

    return model

def robustness_exclude_covid():
    """Reviewer 2: Exclude responses during COVID peak"""
    # We don't actually have dates in this demo but would filter here
    # df = df[~df['response_date'].between('2020-03-01', '2021-06-01')]
    print("Note: Date filtering not implemented in demo")
    pass

def run_all_robustness():
    print("="*60)
    print("Robustness Checks")
    print("="*60)

    print("\n1. Age >= 25")
    robustness_age_restriction()

    print("\n2. Winsorized income")
    robustness_income_winsorize()

    print("\n3. Ordered logit")
    robustness_ordered_logit()

    print("\nResults saved to outputs folder")

if __name__ == '__main__':
    run_all_robustness()
