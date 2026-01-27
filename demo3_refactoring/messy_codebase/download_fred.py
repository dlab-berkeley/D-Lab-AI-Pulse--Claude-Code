# FRED data downloader
# Uses fredapi package
# pip install fredapi

from fredapi import Fred
import pandas as pd
import os

# API key - should be in environment variable
FRED_API_KEY = "your_fred_api_key"

fred = Fred(api_key=FRED_API_KEY)

def download_gdp():
    """Real GDP quarterly"""
    gdp = fred.get_series('GDPC1')
    df = gdp.reset_index()
    df.columns = ['date', 'real_gdp']
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/fred_gdp.csv', index=False)
    print("Downloaded GDP data")
    return df

def download_inflation():
    """CPI for inflation"""
    cpi = fred.get_series('CPIAUCSL')
    df = cpi.reset_index()
    df.columns = ['date', 'cpi']
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/fred_cpi.csv', index=False)
    print("Downloaded CPI data")
    return df

def download_interest_rates():
    """Federal funds rate"""
    ffr = fred.get_series('FEDFUNDS')
    df = ffr.reset_index()
    df.columns = ['date', 'fed_funds_rate']
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/fred_interest.csv', index=False)
    print("Downloaded interest rate data")
    return df

def download_sp500():
    """S&P 500 - added for advisor's suggestion"""
    sp = fred.get_series('SP500')
    df = sp.reset_index()
    df.columns = ['date', 'sp500']
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/fred_sp500.csv', index=False)
    return df

def download_all():
    download_gdp()
    download_inflation()
    download_interest_rates()
    download_sp500()
    print("All FRED data downloaded!")

if __name__ == '__main__':
    download_all()
