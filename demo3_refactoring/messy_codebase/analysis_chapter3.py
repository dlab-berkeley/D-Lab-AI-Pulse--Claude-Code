# Analysis for Chapter 3: Individual-level determinants
# Main regression analysis

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

DATA_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv'
OUTPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/outputs'

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def table1_summary_stats():
    """Generate Table 1: Summary Statistics"""
    df = load_data()

    vars_of_interest = ['satisfaction', 'age', 'female', 'education_years',
                        'income', 'employed', 'trust_index']

    stats_list = []
    for var in vars_of_interest:
        stats_list.append({
            'Variable': var,
            'N': df[var].count(),
            'Mean': df[var].mean(),
            'Std. Dev.': df[var].std(),
            'Min': df[var].min(),
            'Max': df[var].max()
        })

    table1 = pd.DataFrame(stats_list)
    table1.to_csv(f'{OUTPUT_PATH}/table1_summary.csv', index=False)
    print(table1.to_string())
    return table1

def model1_baseline():
    """Model 1: Basic demographics only"""
    df = load_data()

    X = df[['age', 'female', 'education_years', 'income']]
    X = sm.add_constant(X)
    y = df['satisfaction']

    # Drop missing
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust SEs

    with open(f'{OUTPUT_PATH}/model1_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def model2_with_employment():
    """Model 2: Add employment status"""
    df = load_data()

    X = df[['age', 'female', 'education_years', 'income', 'employed']]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    with open(f'{OUTPUT_PATH}/model2_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def model3_with_trust():
    """Model 3: Full model with trust index"""
    df = load_data()

    X = df[['age', 'female', 'education_years', 'income', 'employed', 'trust_index']]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    with open(f'{OUTPUT_PATH}/model3_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def run_all_chapter3():
    print("="*60)
    print("Chapter 3 Analysis")
    print("="*60)

    print("\nTable 1: Summary Statistics")
    table1_summary_stats()

    print("\nModel 1: Baseline")
    m1 = model1_baseline()
    print(f"R-squared: {m1.rsquared:.4f}")

    print("\nModel 2: With Employment")
    m2 = model2_with_employment()
    print(f"R-squared: {m2.rsquared:.4f}")

    print("\nModel 3: Full Model")
    m3 = model3_with_trust()
    print(f"R-squared: {m3.rsquared:.4f}")

if __name__ == '__main__':
    run_all_chapter3()
