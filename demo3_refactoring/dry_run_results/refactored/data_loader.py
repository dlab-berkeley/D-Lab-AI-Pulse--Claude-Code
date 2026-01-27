"""
Data Loader Module - Single Source of Truth for Loading Data.

This module provides a centralized interface for loading all data sources
used in the thesis analysis. It eliminates code duplication and ensures
consistent data loading across all analysis scripts.

Usage:
    from data_loader import DataLoader

    # Load individual datasets
    survey = DataLoader.load_survey_clean()
    census = DataLoader.load_census_data()

    # Load the main analysis dataset
    df = DataLoader.load_analysis_dataset()

    # Load with optional caching
    df = DataLoader.load_analysis_dataset(use_cache=True)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from config import Config


class DataLoader:
    """
    Centralized data loading class for the thesis project.

    All data loading operations should go through this class to ensure
    consistency and reduce code duplication.
    """

    # Cache for loaded dataframes
    _cache: Dict[str, pd.DataFrame] = {}

    # ==========================================================================
    # Raw Data Loaders
    # ==========================================================================

    @classmethod
    def load_census_income(cls) -> pd.DataFrame:
        """
        Load Census income data (median household income by state).

        Returns:
            DataFrame with columns: state_name, median_income, state_fips
        """
        df = pd.read_csv(Config.CENSUS_INCOME_FILE)
        df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
        return df

    @classmethod
    def load_census_education(cls) -> pd.DataFrame:
        """
        Load Census education data (education attainment by state).

        Returns:
            DataFrame with columns: state_name, bachelors, masters,
                                   professional, doctorate, state_fips
        """
        df = pd.read_csv(Config.CENSUS_EDUCATION_FILE)
        edu_cols = ["bachelors", "masters", "professional", "doctorate"]
        for col in edu_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @classmethod
    def load_census_population(cls) -> pd.DataFrame:
        """
        Load Census population data (total population by state).

        Returns:
            DataFrame with columns: state_name, population, state_fips
        """
        df = pd.read_csv(Config.CENSUS_POPULATION_FILE)
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        return df

    @classmethod
    def load_census_data(cls) -> pd.DataFrame:
        """
        Load and merge all Census data sources.

        Combines income, education, and population data.
        Calculates derived variables (pct_bachelors, pct_graduate).

        Returns:
            DataFrame with all Census variables merged by state
        """
        # Load individual Census files
        census_income = cls.load_census_income()
        census_edu = cls.load_census_education()
        census_pop = cls.load_census_population()

        # Merge Census files
        census = census_income.merge(
            census_edu, on=["state_name", "state_fips"], how="outer"
        )
        census = census.merge(
            census_pop, on=["state_name", "state_fips"], how="outer"
        )

        # Calculate derived variables
        census["pct_bachelors"] = (
            census["bachelors"] / census["population"] * 100
        )
        census["pct_graduate"] = (
            (census["masters"] + census["professional"] + census["doctorate"])
            / census["population"]
            * 100
        )

        return census

    @classmethod
    def load_bls_unemployment(cls) -> pd.DataFrame:
        """
        Load BLS unemployment data.

        Returns:
            DataFrame with columns: state, year, period, unemployment_rate
        """
        return pd.read_csv(Config.BLS_UNEMPLOYMENT_FILE)

    @classmethod
    def load_bls_unemployment_annual(
        cls, year: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load BLS unemployment data, filtered to annual averages.

        Args:
            year: Optional year to filter to (e.g., '2023')

        Returns:
            DataFrame with annual average unemployment by state
        """
        df = cls.load_bls_unemployment()

        # M13 is the annual average in BLS data
        df_annual = df[df["period"] == "M13"].copy()

        if year is not None:
            df_annual = df_annual[df_annual["year"] == year]

        return df_annual

    @classmethod
    def load_fred_gdp(cls) -> pd.DataFrame:
        """
        Load FRED Real GDP data.

        Returns:
            DataFrame with columns: date, real_gdp
        """
        df = pd.read_csv(Config.FRED_GDP_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @classmethod
    def load_fred_cpi(cls) -> pd.DataFrame:
        """
        Load FRED CPI (Consumer Price Index) data.

        Returns:
            DataFrame with columns: date, cpi
        """
        df = pd.read_csv(Config.FRED_CPI_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @classmethod
    def load_fred_interest(cls) -> pd.DataFrame:
        """
        Load FRED Federal Funds Rate data.

        Returns:
            DataFrame with columns: date, fed_funds_rate
        """
        df = pd.read_csv(Config.FRED_INTEREST_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @classmethod
    def load_fred_sp500(cls) -> pd.DataFrame:
        """
        Load FRED S&P 500 data.

        Returns:
            DataFrame with columns: date, sp500
        """
        df = pd.read_csv(Config.FRED_SP500_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ==========================================================================
    # Processed Data Loaders
    # ==========================================================================

    @classmethod
    def load_survey_raw(cls) -> pd.DataFrame:
        """
        Load raw survey responses.

        Returns:
            DataFrame with all raw survey responses (including Qualtrics metadata)
        """
        return pd.read_csv(Config.SURVEY_RAW_FILE)

    @classmethod
    def load_survey_clean(cls) -> pd.DataFrame:
        """
        Load cleaned survey data (post-processing).

        Returns:
            DataFrame with cleaned survey variables:
            ResponseId, state, age, female, education_years,
            income, satisfaction, employed, trust_index
        """
        return pd.read_csv(Config.SURVEY_CLEAN_FILE)

    @classmethod
    def load_analysis_dataset(cls, use_cache: bool = False) -> pd.DataFrame:
        """
        Load the main analysis dataset.

        This is the primary dataset used for Chapters 3-4 analysis,
        combining survey data with all contextual variables.

        Args:
            use_cache: If True, return cached version if available

        Returns:
            DataFrame with individual and contextual variables merged
        """
        cache_key = "analysis_dataset"

        if use_cache and cache_key in cls._cache:
            return cls._cache[cache_key].copy()

        df = pd.read_csv(Config.ANALYSIS_DATASET_FILE)

        if use_cache:
            cls._cache[cache_key] = df.copy()

        return df

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the data cache."""
        cls._cache.clear()

    @classmethod
    def check_data_exists(cls, verbose: bool = True) -> Dict[str, bool]:
        """
        Check which data files exist.

        Args:
            verbose: If True, print status for each file

        Returns:
            Dict mapping file names to existence status
        """
        files_to_check = {
            "census_income": Config.CENSUS_INCOME_FILE,
            "census_education": Config.CENSUS_EDUCATION_FILE,
            "census_population": Config.CENSUS_POPULATION_FILE,
            "bls_unemployment": Config.BLS_UNEMPLOYMENT_FILE,
            "fred_gdp": Config.FRED_GDP_FILE,
            "fred_cpi": Config.FRED_CPI_FILE,
            "fred_interest": Config.FRED_INTEREST_FILE,
            "fred_sp500": Config.FRED_SP500_FILE,
            "survey_raw": Config.SURVEY_RAW_FILE,
            "survey_clean": Config.SURVEY_CLEAN_FILE,
            "analysis_dataset": Config.ANALYSIS_DATASET_FILE,
        }

        status = {}
        for name, path in files_to_check.items():
            exists = Path(path).exists()
            status[name] = exists
            if verbose:
                icon = "[OK]" if exists else "[MISSING]"
                print(f"  {icon} {name}: {path}")

        return status

    @classmethod
    def get_dataset_info(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary information about a dataset.

        Args:
            df: DataFrame to summarize

        Returns:
            Dict with dataset statistics
        """
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_pct": (df.isna().sum() / len(df) * 100).to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

    # ==========================================================================
    # Convenience Methods for Analysis
    # ==========================================================================

    @classmethod
    def load_for_regression(
        cls,
        outcome: str = "satisfaction",
        predictors: Optional[List[str]] = None,
        drop_missing: bool = True,
    ) -> tuple:
        """
        Load data prepared for regression analysis.

        Args:
            outcome: Name of the outcome variable
            predictors: List of predictor variable names.
                       If None, uses default set.
            drop_missing: If True, drop rows with missing values

        Returns:
            Tuple of (X, y) where X is predictors DataFrame and y is outcome Series
        """
        if predictors is None:
            predictors = [
                "age",
                "female",
                "education_years",
                "income",
                "employed",
                "trust_index",
            ]

        df = cls.load_analysis_dataset()

        X = df[predictors].copy()
        y = df[outcome].copy()

        if drop_missing:
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

        return X, y

    @classmethod
    def load_by_state(cls) -> pd.DataFrame:
        """
        Load analysis dataset aggregated to state level.

        Returns:
            DataFrame with state-level means and counts
        """
        df = cls.load_analysis_dataset()

        # Numeric columns to aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Aggregate to state level
        state_df = df.groupby("state")[numeric_cols].agg(["mean", "count"])

        # Flatten column names
        state_df.columns = [
            f"{col}_{agg}" for col, agg in state_df.columns
        ]

        return state_df.reset_index()


# ==========================================================================
# Convenience Functions
# ==========================================================================

def load_data(dataset: str = "analysis") -> pd.DataFrame:
    """
    Quick loader function for common datasets.

    Args:
        dataset: One of 'analysis', 'survey', 'census', 'unemployment'

    Returns:
        Requested DataFrame

    Example:
        >>> df = load_data('analysis')
        >>> survey = load_data('survey')
    """
    loaders = {
        "analysis": DataLoader.load_analysis_dataset,
        "survey": DataLoader.load_survey_clean,
        "census": DataLoader.load_census_data,
        "unemployment": DataLoader.load_bls_unemployment,
    }

    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {', '.join(loaders.keys())}"
        )

    return loaders[dataset]()


if __name__ == "__main__":
    # When run directly, check data availability
    print("=" * 70)
    print("DATA LOADER - Checking Data Availability")
    print("=" * 70)
    print("\nChecking data files...")
    status = DataLoader.check_data_exists(verbose=True)

    available = sum(status.values())
    total = len(status)
    print(f"\n{available}/{total} data files found.")

    if available == total:
        print("\nAll data files present. Loading analysis dataset...")
        df = DataLoader.load_analysis_dataset()
        info = DataLoader.get_dataset_info(df)
        print(f"\nAnalysis Dataset Info:")
        print(f"  Rows: {info['n_rows']:,}")
        print(f"  Columns: {info['n_cols']}")
        print(f"  Memory: {info['memory_mb']:.2f} MB")
