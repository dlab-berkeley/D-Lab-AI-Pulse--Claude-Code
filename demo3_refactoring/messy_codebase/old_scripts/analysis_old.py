# OLD VERSION - DO NOT USE
# Keeping for reference only
# Replaced by analysis_chapter3.py

import pandas as pd
from sklearn.linear_model import LinearRegression

def run_regression():
    # Old path
    df = pd.read_csv('/Users/phd_laptop/thesis/data/survey_clean.csv')

    X = df[['age', 'education', 'income']].values
    y = df['satisfaction'].values

    model = LinearRegression()
    model.fit(X, y)

    print("Coefficients:", model.coef_)
    print("R2:", model.score(X, y))

if __name__ == '__main__':
    run_regression()
