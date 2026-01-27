# Download BLS unemployment data
# March 2022

import requests
import pandas as pd
import json

BLS_API_KEY = "your_bls_key"  # Register at bls.gov

def get_unemployment_by_state():
    """Get state-level unemployment rates"""

    # State FIPS to unemployment series mapping
    # This is a subset - full list has all 50 states
    series_map = {
        'LASST010000000000003': 'Alabama',
        'LASST020000000000003': 'Alaska',
        'LASST040000000000003': 'Arizona',
        'LASST060000000000003': 'California',
        # ... more states
    }

    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": list(series_map.keys()),
        "startyear": "2019",
        "endyear": "2023",
        "registrationkey": BLS_API_KEY
    })

    response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/',
                            data=data, headers=headers)

    json_data = response.json()

    # Parse the nested response
    all_data = []
    for series in json_data['Results']['series']:
        series_id = series['seriesID']
        state = series_map.get(series_id, 'Unknown')

        for item in series['data']:
            all_data.append({
                'state': state,
                'year': item['year'],
                'period': item['period'],
                'unemployment_rate': float(item['value'])
            })

    df = pd.DataFrame(all_data)
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/bls_unemployment.csv', index=False)
    print(f"Downloaded {len(df)} unemployment observations")
    return df

def get_employment_by_industry():
    """Added for Chapter 5 analysis"""
    # Similar structure but for industry data
    # TODO: implement this properly
    pass

if __name__ == '__main__':
    get_unemployment_by_state()
