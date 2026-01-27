# Merge all data sources
# This creates the main analysis dataset
# Run AFTER all download and process scripts

import pandas as pd
import numpy as np

# Paths - UPDATE THESE IF MOVING TO NEW COMPUTER
BASE = 'C:/Users/PhDStudent/Desktop/thesis_final'
RAW = f'{BASE}/data_raw'
PROCESSED = f'{BASE}/data_processed'

def merge_survey_with_census():
    """Merge individual survey data with state-level census data"""

    # Load survey
    survey = pd.read_csv(f'{PROCESSED}/survey_clean.csv')

    # Load census data
    census_income = pd.read_csv(f'{RAW}/census_income.csv')
    census_edu = pd.read_csv(f'{RAW}/census_education.csv')
    census_pop = pd.read_csv(f'{RAW}/census_population.csv')

    # Merge census files first
    census = census_income.merge(census_edu, on=['state_name', 'state_fips'])
    census = census.merge(census_pop, on=['state_name', 'state_fips'])

    # Calculate additional variables
    census['pct_bachelors'] = census['bachelors'] / census['population'] * 100
    census['pct_graduate'] = (census['masters'] + census['professional'] + census['doctorate']) / census['population'] * 100

    # Rename for merge
    census = census.rename(columns={'state_name': 'state'})

    # Merge with survey
    merged = survey.merge(census, on='state', how='left')

    return merged

def add_macro_data(df):
    """Add time-varying macro data based on survey date"""
    # This is tricky because survey spans multiple months

    # Load FRED data
    gdp = pd.read_csv(f'{RAW}/fred_gdp.csv')
    cpi = pd.read_csv(f'{RAW}/fred_cpi.csv')

    # For simplicity, use most recent values
    # TODO: properly match by survey response date
    df['gdp'] = gdp['real_gdp'].iloc[-1]
    df['cpi'] = cpi['cpi'].iloc[-1]

    return df

def add_unemployment(df):
    """Add state-level unemployment from BLS"""
    unemp = pd.read_csv(f'{RAW}/bls_unemployment.csv')

    # Get most recent annual average by state
    unemp_annual = unemp[unemp['period'] == 'M13']  # M13 is annual average
    unemp_recent = unemp_annual[unemp_annual['year'] == '2023']

    df = df.merge(unemp_recent[['state', 'unemployment_rate']], on='state', how='left')

    return df

def create_final_dataset():
    df = merge_survey_with_census()
    df = add_macro_data(df)
    df = add_unemployment(df)

    # Save
    df.to_csv(f'{PROCESSED}/analysis_dataset.csv', index=False)
    print(f"Final dataset: {len(df)} observations, {len(df.columns)} variables")

    # Also save summary stats
    df.describe().to_csv(f'{BASE}/outputs/summary_stats.csv')

    return df

if __name__ == '__main__':
    create_final_dataset()
