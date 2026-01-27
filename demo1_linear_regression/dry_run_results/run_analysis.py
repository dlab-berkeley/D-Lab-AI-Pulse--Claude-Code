"""
Productivity Study Analysis

This script analyzes the relationship between coffee consumption
and productivity using the LinearRegression class.

Author: AI-Pulse Workshop
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import the LinearRegression class from the local module
from linear_regression import LinearRegression


def main():
    # Define paths
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "productivity_study.csv"
    output_plot_path = script_dir / "productivity_regression_plot.png"

    # Step 1: Load the data
    print("=" * 60)
    print("Loading Data")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} observations")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDescriptive Statistics:")
    print(df.describe())

    # Step 2: Prepare data for regression
    # Predict Productivity_Score from Coffee_Cups
    X = df["Coffee_Cups"].values.reshape(-1, 1)
    y = df["Productivity_Score"].values

    # Step 3: Fit the linear regression model
    print("\n" + "=" * 60)
    print("Fitting Linear Regression Model")
    print("=" * 60)
    print("\nPredicting Productivity_Score from Coffee_Cups\n")

    model = LinearRegression()
    model.fit(X, y, feature_names=["Coffee_Cups"])

    # Step 4: Print the model summary
    print(model.summary())

    # Additional insights
    print("\n" + "=" * 60)
    print("Model Interpretation")
    print("=" * 60)
    print(f"\nIntercept: {model.intercept:.4f}")
    print(f"Coefficient (Coffee_Cups): {model.coefficients[0]:.4f}")
    print(f"R-squared: {model.r_squared:.4f}")
    print(f"\nInterpretation:")
    print(f"  - For each additional cup of coffee, productivity score")
    print(f"    increases by approximately {model.coefficients[0]:.2f} points.")
    print(f"  - The model explains {model.r_squared * 100:.1f}% of the variance")
    print(f"    in productivity scores.")

    # Step 5: Save the regression plot
    print("\n" + "=" * 60)
    print("Generating Regression Plot")
    print("=" * 60)

    fig = model.plot_fit(
        X, y,
        title="Coffee Consumption vs Productivity Score",
        show_residuals=True,
        figsize=(12, 5)
    )

    # Customize labels
    fig.axes[0].set_xlabel("Coffee Cups", fontsize=12)
    fig.axes[0].set_ylabel("Productivity Score", fontsize=12)

    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_plot_path}")

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
