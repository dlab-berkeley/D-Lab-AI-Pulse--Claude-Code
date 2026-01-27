"""
Configuration module for thesis analysis project.

All paths are relative to the project root using pathlib.
API keys are loaded from environment variables with clear placeholders.

Usage:
    from config import Config

    # Access paths
    Config.RAW_DATA_DIR
    Config.PROCESSED_DATA_DIR

    # Access API keys
    Config.CENSUS_API_KEY
"""

import os
from pathlib import Path


class Config:
    """Central configuration for the thesis analysis project."""

    # ==========================================================================
    # Project Root
    # ==========================================================================
    # Automatically detect project root (directory containing this config file)
    PROJECT_ROOT = Path(__file__).parent.resolve()

    # ==========================================================================
    # Directory Structure
    # ==========================================================================
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    FIGURES_DIR = PROJECT_ROOT / "figures"
    TABLES_DIR = PROJECT_ROOT / "tables"

    # ==========================================================================
    # Data File Paths - Raw Data
    # ==========================================================================
    # Census data
    CENSUS_INCOME_FILE = RAW_DATA_DIR / "census_income.csv"
    CENSUS_EDUCATION_FILE = RAW_DATA_DIR / "census_education.csv"
    CENSUS_POPULATION_FILE = RAW_DATA_DIR / "census_population.csv"

    # BLS data
    BLS_UNEMPLOYMENT_FILE = RAW_DATA_DIR / "bls_unemployment.csv"

    # FRED data
    FRED_GDP_FILE = RAW_DATA_DIR / "fred_gdp.csv"
    FRED_CPI_FILE = RAW_DATA_DIR / "fred_cpi.csv"
    FRED_INTEREST_FILE = RAW_DATA_DIR / "fred_interest.csv"
    FRED_SP500_FILE = RAW_DATA_DIR / "fred_sp500.csv"

    # Survey data
    SURVEY_RAW_FILE = RAW_DATA_DIR / "survey_responses_raw.csv"

    # ==========================================================================
    # Data File Paths - Processed Data
    # ==========================================================================
    SURVEY_CLEAN_FILE = PROCESSED_DATA_DIR / "survey_clean.csv"
    ANALYSIS_DATASET_FILE = PROCESSED_DATA_DIR / "analysis_dataset.csv"

    # ==========================================================================
    # Output File Paths
    # ==========================================================================
    SUMMARY_STATS_FILE = OUTPUT_DIR / "summary_stats.csv"

    # ==========================================================================
    # API Keys - Loaded from Environment Variables
    # ==========================================================================
    # Set these environment variables before running:
    #   export CENSUS_API_KEY="your_key_here"
    #   export BLS_API_KEY="your_key_here"
    #   export FRED_API_KEY="your_key_here"
    #   export QUALTRICS_API_TOKEN="your_token_here"

    CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")
    BLS_API_KEY = os.environ.get("BLS_API_KEY", "")
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
    QUALTRICS_API_TOKEN = os.environ.get("QUALTRICS_API_TOKEN", "")

    # ==========================================================================
    # API Endpoints
    # ==========================================================================
    CENSUS_BASE_URL = "https://api.census.gov/data/2021/acs/acs5"
    BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    # Qualtrics configuration
    QUALTRICS_DATACENTER = os.environ.get("QUALTRICS_DATACENTER", "ca1")
    QUALTRICS_SURVEY_ID = os.environ.get("QUALTRICS_SURVEY_ID", "")

    # ==========================================================================
    # Analysis Parameters
    # ==========================================================================
    # Year ranges for data downloads
    BLS_START_YEAR = "2019"
    BLS_END_YEAR = "2023"

    # Winsorization limits for robustness checks
    WINSORIZE_LOWER = 0.01
    WINSORIZE_UPPER = 0.99

    # Minimum age for robustness check
    MIN_AGE_ROBUSTNESS = 25

    # ==========================================================================
    # Class Methods
    # ==========================================================================
    @classmethod
    def ensure_directories_exist(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.FIGURES_DIR,
            cls.TABLES_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_api_keys(cls, required_keys=None):
        """
        Validate that required API keys are set.

        Args:
            required_keys: List of key names to validate.
                          If None, validates all keys.

        Returns:
            dict: Status of each key (True if set, False if empty)

        Raises:
            ValueError: If any required key is missing
        """
        all_keys = {
            "CENSUS_API_KEY": cls.CENSUS_API_KEY,
            "BLS_API_KEY": cls.BLS_API_KEY,
            "FRED_API_KEY": cls.FRED_API_KEY,
            "QUALTRICS_API_TOKEN": cls.QUALTRICS_API_TOKEN,
        }

        if required_keys is None:
            required_keys = list(all_keys.keys())

        status = {}
        missing = []

        for key in required_keys:
            if key in all_keys:
                is_set = bool(all_keys[key])
                status[key] = is_set
                if not is_set:
                    missing.append(key)

        if missing:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing)}. "
                f"Please set these environment variables before running."
            )

        return status

    @classmethod
    def get_qualtrics_base_url(cls):
        """Construct the Qualtrics API base URL."""
        return (
            f"https://{cls.QUALTRICS_DATACENTER}.qualtrics.com/API/v3/"
            f"surveys/{cls.QUALTRICS_SURVEY_ID}/export-responses"
        )

    @classmethod
    def print_config(cls):
        """Print current configuration for debugging."""
        print("=" * 70)
        print("PROJECT CONFIGURATION")
        print("=" * 70)
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"\nDirectories:")
        print(f"  Raw Data:       {cls.RAW_DATA_DIR}")
        print(f"  Processed Data: {cls.PROCESSED_DATA_DIR}")
        print(f"  Outputs:        {cls.OUTPUT_DIR}")
        print(f"  Figures:        {cls.FIGURES_DIR}")
        print(f"  Tables:         {cls.TABLES_DIR}")
        print(f"\nAPI Keys Set:")
        print(f"  CENSUS_API_KEY:      {'Yes' if cls.CENSUS_API_KEY else 'No'}")
        print(f"  BLS_API_KEY:         {'Yes' if cls.BLS_API_KEY else 'No'}")
        print(f"  FRED_API_KEY:        {'Yes' if cls.FRED_API_KEY else 'No'}")
        print(f"  QUALTRICS_API_TOKEN: {'Yes' if cls.QUALTRICS_API_TOKEN else 'No'}")
        print("=" * 70)


# Convenience function for quick path access
def get_path(name: str) -> Path:
    """
    Get a configured path by name.

    Args:
        name: Name of the path (e.g., 'raw_data', 'processed_data', 'output')

    Returns:
        Path object for the requested location

    Example:
        >>> get_path('raw_data')
        PosixPath('/path/to/project/data/raw')
    """
    path_map = {
        "root": Config.PROJECT_ROOT,
        "data": Config.DATA_DIR,
        "raw_data": Config.RAW_DATA_DIR,
        "processed_data": Config.PROCESSED_DATA_DIR,
        "output": Config.OUTPUT_DIR,
        "figures": Config.FIGURES_DIR,
        "tables": Config.TABLES_DIR,
    }

    if name not in path_map:
        raise KeyError(
            f"Unknown path name: {name}. "
            f"Available paths: {', '.join(path_map.keys())}"
        )

    return path_map[name]


if __name__ == "__main__":
    # Print configuration when run directly
    Config.print_config()

    # Optionally create directories
    print("\nCreating directories...")
    Config.ensure_directories_exist()
    print("Done!")
