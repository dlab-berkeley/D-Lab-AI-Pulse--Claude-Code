"""
Validation Script: Compare Custom LinearRegression vs sklearn

This script validates our custom LinearRegression implementation by comparing
its outputs against sklearn's LinearRegression on the same dataset.

Author: AI-Pulse Workshop
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import our custom implementation
from linear_regression import LinearRegression as CustomLinearRegression

# Import sklearn's implementation
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score


def compare_implementations(tolerance: float = 1e-10):
    """
    Compare custom LinearRegression against sklearn's implementation.

    Parameters
    ----------
    tolerance : float
        Maximum allowed difference for floating point comparisons.

    Returns
    -------
    bool
        True if all comparisons pass, False otherwise.
    """
    # Define paths
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "productivity_study.csv"

    print("=" * 70)
    print("   Validation: Custom LinearRegression vs sklearn LinearRegression")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    X = df["Coffee_Cups"].values.reshape(-1, 1)
    y = df["Productivity_Score"].values
    print(f"  Dataset: {data_path.name}")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {X.shape[1]}")
    print()

    # Fit our custom model
    print("Fitting Custom LinearRegression...")
    custom_model = CustomLinearRegression()
    custom_model.fit(X, y, feature_names=["Coffee_Cups"])

    custom_intercept = custom_model.intercept
    custom_coefs = custom_model.coefficients
    custom_r2 = custom_model.r_squared
    custom_predictions = custom_model.predict(X)

    print(f"  Intercept:     {custom_intercept:.10f}")
    print(f"  Coefficient:   {custom_coefs[0]:.10f}")
    print(f"  R-squared:     {custom_r2:.10f}")
    print()

    # Fit sklearn model
    print("Fitting sklearn LinearRegression...")
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X, y)

    sklearn_intercept = sklearn_model.intercept_
    sklearn_coefs = sklearn_model.coef_
    sklearn_predictions = sklearn_model.predict(X)
    sklearn_r2 = r2_score(y, sklearn_predictions)

    print(f"  Intercept:     {sklearn_intercept:.10f}")
    print(f"  Coefficient:   {sklearn_coefs[0]:.10f}")
    print(f"  R-squared:     {sklearn_r2:.10f}")
    print()

    # Compare results
    print("-" * 70)
    print("COMPARISON RESULTS")
    print("-" * 70)
    print()

    all_passed = True

    # Compare intercepts
    intercept_diff = abs(custom_intercept - sklearn_intercept)
    intercept_match = intercept_diff <= tolerance
    status = "PASS" if intercept_match else "FAIL"
    print(f"Intercept Comparison:")
    print(f"  Custom:     {custom_intercept:.15f}")
    print(f"  sklearn:    {sklearn_intercept:.15f}")
    print(f"  Difference: {intercept_diff:.2e}")
    print(f"  Status:     [{status}]")
    print()
    if not intercept_match:
        all_passed = False

    # Compare coefficients
    coef_diff = abs(custom_coefs[0] - sklearn_coefs[0])
    coef_match = coef_diff <= tolerance
    status = "PASS" if coef_match else "FAIL"
    print(f"Coefficient Comparison:")
    print(f"  Custom:     {custom_coefs[0]:.15f}")
    print(f"  sklearn:    {sklearn_coefs[0]:.15f}")
    print(f"  Difference: {coef_diff:.2e}")
    print(f"  Status:     [{status}]")
    print()
    if not coef_match:
        all_passed = False

    # Compare R-squared
    r2_diff = abs(custom_r2 - sklearn_r2)
    r2_match = r2_diff <= tolerance
    status = "PASS" if r2_match else "FAIL"
    print(f"R-squared Comparison:")
    print(f"  Custom:     {custom_r2:.15f}")
    print(f"  sklearn:    {sklearn_r2:.15f}")
    print(f"  Difference: {r2_diff:.2e}")
    print(f"  Status:     [{status}]")
    print()
    if not r2_match:
        all_passed = False

    # Compare predictions (using max absolute difference)
    pred_diff = np.max(np.abs(custom_predictions - sklearn_predictions))
    pred_match = pred_diff <= tolerance
    status = "PASS" if pred_match else "FAIL"
    print(f"Predictions Comparison:")
    print(f"  Max Absolute Difference: {pred_diff:.2e}")
    print(f"  Status:     [{status}]")
    print()
    if not pred_match:
        all_passed = False

    # Final summary
    print("=" * 70)
    if all_passed:
        print("VALIDATION RESULT: ALL TESTS PASSED")
        print(f"Our implementation matches sklearn within tolerance ({tolerance:.0e})")
    else:
        print("VALIDATION RESULT: SOME TESTS FAILED")
        print("Review the differences above for details.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    # Run validation with default tolerance
    success = compare_implementations(tolerance=1e-10)

    # Exit with appropriate code
    exit(0 if success else 1)
