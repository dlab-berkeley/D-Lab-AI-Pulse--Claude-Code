"""
Linear Regression Implementation using Ordinary Least Squares (OLS)

This module provides a LinearRegression class that implements OLS regression
from scratch using the closed-form solution (normal equation).

Author: AI-Pulse Workshop
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Optional, Union


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression.

    This class implements linear regression using the closed-form OLS solution
    (normal equation): beta = (X'X)^(-1) X'y

    Attributes
    ----------
    coefficients : np.ndarray
        The regression coefficients (excluding intercept).
    intercept : float
        The intercept (bias) term.
    r_squared : float
        The coefficient of determination R^2.
    standard_errors : np.ndarray
        Standard errors of the coefficients (including intercept).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    >>> model.predict(np.array([[6]]))
    array([11.98])
    """

    def __init__(self):
        """Initialize the LinearRegression model."""
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: Optional[float] = None
        self._r_squared: Optional[float] = None
        self._adj_r_squared: Optional[float] = None
        self._standard_errors: Optional[np.ndarray] = None
        self._t_statistics: Optional[np.ndarray] = None
        self._p_values: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._fitted_values: Optional[np.ndarray] = None
        self._n_samples: Optional[int] = None
        self._n_features: Optional[int] = None
        self._feature_names: Optional[list] = None
        self._mse: Optional[float] = None
        self._f_statistic: Optional[float] = None
        self._f_pvalue: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[list] = None) -> 'LinearRegression':
        """
        Fit the linear regression model using OLS.

        Uses the closed-form solution (normal equation):
        beta = (X'X)^(-1) X'y

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
            Can also be 1D array of shape (n_samples,) for simple regression.
        y : np.ndarray
            Target values of shape (n_samples,).
        feature_names : list, optional
            Names of the features for the summary table.

        Returns
        -------
        self : LinearRegression
            The fitted model.

        Raises
        ------
        ValueError
            If X and y have incompatible shapes.
        """
        # Ensure X is 2D
        X = np.atleast_2d(X)
        if X.shape[0] == 1 and len(y) > 1:
            X = X.T

        y = np.asarray(y).flatten()

        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y have incompatible shapes: X has {X.shape[0]} samples, "
                f"y has {len(y)} samples."
            )

        self._n_samples, self._n_features = X.shape
        self._feature_names = feature_names or [f"X{i+1}" for i in range(self._n_features)]

        # Add column of ones for intercept (design matrix)
        X_design = np.column_stack([np.ones(self._n_samples), X])

        # OLS closed-form solution: beta = (X'X)^(-1) X'y
        # Using pseudo-inverse for numerical stability
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ X_design.T @ y

        # Extract intercept and coefficients
        self._intercept = beta[0]
        self._coefficients = beta[1:]

        # Calculate fitted values and residuals
        self._fitted_values = X_design @ beta
        self._residuals = y - self._fitted_values

        # Calculate R-squared
        ss_res = np.sum(self._residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self._r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Adjusted R-squared
        n, p = self._n_samples, self._n_features
        if n - p - 1 > 0:
            self._adj_r_squared = 1 - (1 - self._r_squared) * (n - 1) / (n - p - 1)
        else:
            self._adj_r_squared = self._r_squared

        # Mean Squared Error (unbiased estimate of variance)
        dof = n - p - 1  # degrees of freedom
        self._mse = ss_res / dof if dof > 0 else ss_res

        # Standard errors of coefficients
        # Var(beta) = sigma^2 * (X'X)^(-1)
        var_beta = self._mse * XtX_inv
        self._standard_errors = np.sqrt(np.diag(var_beta))

        # T-statistics
        self._t_statistics = beta / self._standard_errors

        # P-values (two-tailed test)
        self._p_values = 2 * (1 - stats.t.cdf(np.abs(self._t_statistics), dof))

        # F-statistic for overall model significance
        if p > 0 and dof > 0:
            ss_reg = ss_tot - ss_res
            ms_reg = ss_reg / p
            ms_res = ss_res / dof
            self._f_statistic = ms_reg / ms_res if ms_res > 0 else np.inf
            self._f_pvalue = 1 - stats.f.cdf(self._f_statistic, p, dof)
        else:
            self._f_statistic = None
            self._f_pvalue = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Samples of shape (n_samples, n_features) or (n_features,) for single sample.

        Returns
        -------
        y_pred : np.ndarray
            Predicted values of shape (n_samples,).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If X has incorrect number of features.
        """
        if self._coefficients is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.atleast_2d(X)
        if X.shape[0] == 1 and X.shape[1] != self._n_features:
            X = X.T

        if X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained with "
                f"{self._n_features} features."
            )

        return self._intercept + X @ self._coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """
        Get the regression coefficients (excluding intercept).

        Returns
        -------
        np.ndarray
            Array of coefficients.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._coefficients is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._coefficients.copy()

    @property
    def intercept(self) -> float:
        """
        Get the intercept (bias) term.

        Returns
        -------
        float
            The intercept value.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._intercept is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._intercept

    @property
    def r_squared(self) -> float:
        """
        Get the coefficient of determination R^2.

        R^2 measures the proportion of variance in the dependent variable
        that is predictable from the independent variable(s).

        Returns
        -------
        float
            R-squared value between 0 and 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._r_squared is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._r_squared

    @property
    def standard_errors(self) -> np.ndarray:
        """
        Get the standard errors of the coefficients.

        Returns array with standard error of intercept first,
        followed by standard errors of coefficients.

        Returns
        -------
        np.ndarray
            Array of standard errors [SE(intercept), SE(coef1), SE(coef2), ...].

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._standard_errors is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._standard_errors.copy()

    @property
    def residuals(self) -> np.ndarray:
        """
        Get the residuals from the fitted model.

        Returns
        -------
        np.ndarray
            Array of residuals (y - y_pred).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._residuals is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._residuals.copy()

    def summary(self) -> str:
        """
        Generate a formatted summary table of the regression results.

        Similar to statsmodels OLS summary output, includes:
        - Model statistics (R-squared, F-statistic, etc.)
        - Coefficient table with standard errors, t-stats, and p-values

        Returns
        -------
        str
            Formatted summary string.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._coefficients is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # Build the summary string
        lines = []
        lines.append("=" * 78)
        lines.append(" " * 25 + "OLS Regression Results")
        lines.append("=" * 78)

        # Model statistics
        lines.append(f"{'Dep. Variable:':<20} y" + " " * 15 +
                    f"{'R-squared:':<15} {self._r_squared:.6f}")
        lines.append(f"{'Model:':<20} OLS" + " " * 13 +
                    f"{'Adj. R-squared:':<15} {self._adj_r_squared:.6f}")
        lines.append(f"{'No. Observations:':<20} {self._n_samples}" + " " * (16 - len(str(self._n_samples))) +
                    f"{'F-statistic:':<15} {self._f_statistic:.4f}" if self._f_statistic else "")
        lines.append(f"{'Df Residuals:':<20} {self._n_samples - self._n_features - 1}" +
                    " " * (16 - len(str(self._n_samples - self._n_features - 1))) +
                    f"{'Prob (F-statistic):':<15} {self._f_pvalue:.4e}" if self._f_pvalue else "")
        lines.append(f"{'Df Model:':<20} {self._n_features}")

        lines.append("-" * 78)

        # Coefficient table header
        header = f"{'Variable':<15} {'Coefficient':>12} {'Std. Error':>12} {'t-stat':>10} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        lines.append(header)
        lines.append("-" * 78)

        # Degrees of freedom for confidence intervals
        dof = self._n_samples - self._n_features - 1
        t_crit = stats.t.ppf(0.975, dof) if dof > 0 else 1.96

        # Intercept row
        coef = self._intercept
        se = self._standard_errors[0]
        t_stat = self._t_statistics[0]
        p_val = self._p_values[0]
        ci_low = coef - t_crit * se
        ci_high = coef + t_crit * se

        lines.append(
            f"{'const':<15} {coef:>12.6f} {se:>12.6f} {t_stat:>10.3f} "
            f"{p_val:>10.3f} {ci_low:>10.3f} {ci_high:>10.3f}"
        )

        # Coefficient rows
        for i, name in enumerate(self._feature_names):
            coef = self._coefficients[i]
            se = self._standard_errors[i + 1]
            t_stat = self._t_statistics[i + 1]
            p_val = self._p_values[i + 1]
            ci_low = coef - t_crit * se
            ci_high = coef + t_crit * se

            # Truncate long names
            display_name = name[:14] if len(name) > 14 else name

            lines.append(
                f"{display_name:<15} {coef:>12.6f} {se:>12.6f} {t_stat:>10.3f} "
                f"{p_val:>10.3f} {ci_low:>10.3f} {ci_high:>10.3f}"
            )

        lines.append("=" * 78)

        # Add notes
        lines.append("Notes:")
        lines.append(f"[1] Standard Errors assume that the covariance matrix is correctly specified.")
        lines.append(f"[2] MSE (Root): {np.sqrt(self._mse):.6f}")

        return "\n".join(lines)

    def plot_fit(self, X: np.ndarray, y: np.ndarray,
                 feature_index: int = 0,
                 figsize: tuple = (10, 6),
                 show_residuals: bool = False,
                 title: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization of the regression fit.

        For simple regression (1 feature), plots the data points and fitted line.
        For multiple regression, plots against one selected feature.

        Parameters
        ----------
        X : np.ndarray
            Feature data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        feature_index : int, default=0
            Index of feature to plot on x-axis (for multiple regression).
        figsize : tuple, default=(10, 6)
            Figure size in inches.
        show_residuals : bool, default=False
            If True, creates a 2-panel plot with residuals.
        title : str, optional
            Custom title for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._coefficients is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.atleast_2d(X)
        if X.shape[0] == 1 and len(y) > 1:
            X = X.T
        y = np.asarray(y).flatten()

        # Get predictions
        y_pred = self.predict(X)

        # Select feature for x-axis
        x_plot = X[:, feature_index]
        feature_name = (self._feature_names[feature_index]
                       if self._feature_names else f"X{feature_index + 1}")

        if show_residuals:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = axes
        else:
            fig, ax1 = plt.subplots(figsize=figsize)

        # Main scatter plot with regression line
        ax1.scatter(x_plot, y, alpha=0.6, label='Observed', color='steelblue',
                   edgecolors='white', linewidth=0.5, s=60)

        # Sort for line plot
        sort_idx = np.argsort(x_plot)
        ax1.plot(x_plot[sort_idx], y_pred[sort_idx], color='crimson',
                linewidth=2, label='Fitted line')

        # Add confidence band (approximate)
        residual_std = np.std(y - y_pred)
        ax1.fill_between(x_plot[sort_idx],
                        y_pred[sort_idx] - 1.96 * residual_std,
                        y_pred[sort_idx] + 1.96 * residual_std,
                        alpha=0.2, color='crimson', label='95% CI')

        ax1.set_xlabel(feature_name, fontsize=11)
        ax1.set_ylabel('y', fontsize=11)

        if title:
            ax1.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax1.set_title(f'Linear Regression Fit (R² = {self._r_squared:.4f})',
                         fontsize=12, fontweight='bold')

        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Add equation annotation
        if self._n_features == 1:
            eq_text = f'y = {self._intercept:.3f} + {self._coefficients[0]:.3f}x'
        else:
            eq_text = f'y = {self._intercept:.3f} + ... (multiple features)'

        ax1.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Residual plot
        if show_residuals:
            residuals = y - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color='steelblue',
                       edgecolors='white', linewidth=0.5, s=60)
            ax2.axhline(y=0, color='crimson', linestyle='--', linewidth=1.5)
            ax2.set_xlabel('Fitted Values', fontsize=11)
            ax2.set_ylabel('Residuals', fontsize=11)
            ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # Add horizontal bands at +/- 2 std
            ax2.axhline(y=2*residual_std, color='gray', linestyle=':', alpha=0.5)
            ax2.axhline(y=-2*residual_std, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()
        return fig


# Demonstration code
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate sample data
    n_samples = 100
    X = np.random.uniform(0, 10, (n_samples, 2))
    true_intercept = 2.0
    true_coefs = np.array([3.0, -1.5])
    noise = np.random.normal(0, 1.5, n_samples)
    y = true_intercept + X @ true_coefs + noise

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y, feature_names=['Feature1', 'Feature2'])

    # Print summary
    print(model.summary())
    print()

    # Print properties
    print("Model Properties:")
    print(f"  Intercept: {model.intercept:.4f}")
    print(f"  Coefficients: {model.coefficients}")
    print(f"  R-squared: {model.r_squared:.4f}")
    print(f"  Standard Errors: {model.standard_errors}")
    print()

    # Make predictions
    X_new = np.array([[5.0, 3.0], [7.5, 1.0]])
    predictions = model.predict(X_new)
    print(f"Predictions for {X_new.tolist()}:")
    print(f"  {predictions}")
    print()

    # Simple regression example
    print("\n" + "="*50)
    print("Simple Regression Example")
    print("="*50 + "\n")

    X_simple = np.random.uniform(0, 10, (50, 1))
    y_simple = 2.5 + 1.8 * X_simple.flatten() + np.random.normal(0, 1, 50)

    simple_model = LinearRegression()
    simple_model.fit(X_simple, y_simple, feature_names=['X'])
    print(simple_model.summary())

    # Create visualization
    fig = simple_model.plot_fit(X_simple, y_simple, show_residuals=True)
    plt.savefig('/mnt/c/Users/Bruno/OneDrive/Desktop/Projects/D-Lab/AI-Pulse Claude Code/Claude Code Workshop/demo1_linear_regression/dry_run_results/regression_plot.png',
                dpi=150, bbox_inches='tight')
    print("\nPlot saved to 'regression_plot.png'")
    plt.show()
