# Process survey data
# Clean and prepare for analysis
# Updated multiple times...

import pandas as pd
import numpy as np

INPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_raw/survey_responses_raw.csv'
OUTPUT_PATH = 'C:/Users/PhDStudent/Desktop/thesis_final/data_processed/survey_clean.csv'

def load_raw():
    df = pd.read_csv(INPUT_PATH)
    # Skip first two rows (Qualtrics metadata)
    df = df.iloc[2:]
    return df

def clean_demographics(df):
    """Clean demographic variables"""
    # Age
    df['age'] = pd.to_numeric(df['Q2_age'], errors='coerce')
    df = df[df['age'].between(18, 100)]

    # Gender
    df['female'] = (df['Q3_gender'] == 'Female').astype(int)

    # Education (recode to years)
    edu_map = {
        'Less than high school': 10,
        'High school': 12,
        'Some college': 14,
        'Bachelor\'s degree': 16,
        'Master\'s degree': 18,
        'Doctoral degree': 22
    }
    df['education_years'] = df['Q4_education'].map(edu_map)

    # Income (take midpoint of brackets)
    income_map = {
        'Less than $25,000': 12500,
        '$25,000 - $49,999': 37500,
        '$50,000 - $74,999': 62500,
        '$75,000 - $99,999': 87500,
        '$100,000 - $149,999': 125000,
        '$150,000 or more': 175000
    }
    df['income'] = df['Q5_income'].map(income_map)

    return df

def clean_outcomes(df):
    """Clean outcome variables"""
    # Life satisfaction (1-10 scale)
    df['satisfaction'] = pd.to_numeric(df['Q10_satisfaction'], errors='coerce')

    # Employment status
    df['employed'] = (df['Q6_employment'] == 'Employed full-time').astype(int)

    # Trust in institutions (average of multiple items)
    trust_cols = ['Q15_trust_gov', 'Q16_trust_media', 'Q17_trust_science']
    for col in trust_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['trust_index'] = df[trust_cols].mean(axis=1)

    return df

def clean_state(df):
    """Standardize state names for merging"""
    # Some responses have abbreviations, some have full names
    state_abbrev = {
        'CA': 'California', 'TX': 'Texas', 'NY': 'New York',
        'FL': 'Florida', 'IL': 'Illinois', 'PA': 'Pennsylvania',
        # ... etc
    }
    df['state'] = df['Q1_state'].replace(state_abbrev)
    return df

def process_all():
    df = load_raw()
    df = clean_demographics(df)
    df = clean_outcomes(df)
    df = clean_state(df)

    # Keep only needed columns
    keep_cols = ['ResponseId', 'state', 'age', 'female', 'education_years',
                 'income', 'satisfaction', 'employed', 'trust_index']
    df = df[keep_cols]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Processed {len(df)} responses, saved to {OUTPUT_PATH}")
    return df

if __name__ == '__main__':
    process_all()
