# Download Census ACS data
# Written: Feb 2022
# Note: Need to get API key from census.gov

import requests
import pandas as pd
import os

CENSUS_API_KEY = "your_api_key_here"  # Get from environment in production
BASE_URL = "https://api.census.gov/data/2021/acs/acs5"

def download_income_data():
    """Download median household income by state"""
    params = {
        'get': 'NAME,B19013_001E',  # Median household income
        'for': 'state:*',
        'key': CENSUS_API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = ['state_name', 'median_income', 'state_fips']
    df['median_income'] = pd.to_numeric(df['median_income'])

    # Save to raw data folder
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/census_income.csv', index=False)
    print(f"Saved {len(df)} rows to census_income.csv")
    return df

def download_education_data():
    """Download education attainment by state"""
    params = {
        'get': 'NAME,B15003_022E,B15003_023E,B15003_024E,B15003_025E',  # BA, MA, Prof, PhD
        'for': 'state:*',
        'key': CENSUS_API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = ['state_name', 'bachelors', 'masters', 'professional', 'doctorate', 'state_fips']

    for col in ['bachelors', 'masters', 'professional', 'doctorate']:
        df[col] = pd.to_numeric(df[col])

    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/census_education.csv', index=False)
    print(f"Saved {len(df)} rows to census_education.csv")
    return df

def download_population_data():
    """Added later for robustness checks"""
    params = {
        'get': 'NAME,B01003_001E',  # Total population
        'for': 'state:*',
        'key': CENSUS_API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = ['state_name', 'population', 'state_fips']
    df['population'] = pd.to_numeric(df['population'])

    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/census_population.csv', index=False)
    return df

if __name__ == '__main__':
    download_income_data()
    download_education_data()
    # download_population_data()  # uncomment if needed
