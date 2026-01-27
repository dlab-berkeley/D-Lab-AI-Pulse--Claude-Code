# Analysis for Chapter 4: Contextual effects
# Multilevel models and state-level predictors

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/analysis_dataset.csv'
OUTPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/outputs'

def load_data():
    # Same function as chapter 3 but copied here
    df = pd.read_csv(DATA_PATH)
    return df

def model4_state_fixed_effects():
    """Add state fixed effects"""
    df = load_data()

    # Create state dummies
    df = pd.get_dummies(df, columns=['state'], drop_first=True)

    # Individual vars
    ind_vars = ['age', 'female', 'education_years', 'income', 'employed', 'trust_index']
    state_vars = [c for c in df.columns if c.startswith('state_')]

    X = df[ind_vars + state_vars]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='HC3')

    # Save full results
    with open(f'{OUTPUT_PATH}/model4_fe_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def model5_state_predictors():
    """Add state-level predictors instead of FE"""
    df = load_data()

    # Use state-level vars from census
    vars = ['age', 'female', 'education_years', 'income', 'employed', 'trust_index',
            'median_income', 'pct_bachelors', 'unemployment_rate']

    X = df[vars]
    X = sm.add_constant(X)
    y = df['satisfaction']

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df.loc[mask.values, 'state_fips']})

    with open(f'{OUTPUT_PATH}/model5_context_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def model6_multilevel():
    """Random intercept model - individuals nested in states"""
    df = load_data()

    # Using statsmodels MixedLM
    formula = 'satisfaction ~ age + female + education_years + income + employed + trust_index'

    # Need to drop NAs first
    vars = ['satisfaction', 'age', 'female', 'education_years', 'income', 'employed', 'trust_index', 'state']
    df_clean = df[vars].dropna()

    model = smf.mixedlm(formula, df_clean, groups=df_clean['state']).fit()

    with open(f'{OUTPUT_PATH}/model6_multilevel_results.txt', 'w') as f:
        f.write(model.summary().as_text())

    return model

def run_all_chapter4():
    print("="*60)
    print("Chapter 4 Analysis")
    print("="*60)

    print("\nModel 4: State Fixed Effects")
    m4 = model4_state_fixed_effects()
    print(f"R-squared: {m4.rsquared:.4f}")

    print("\nModel 5: State-level Predictors")
    m5 = model5_state_predictors()
    print(f"R-squared: {m5.rsquared:.4f}")

    print("\nModel 6: Multilevel Model")
    m6 = model6_multilevel()

if __name__ == '__main__':
    run_all_chapter4()
