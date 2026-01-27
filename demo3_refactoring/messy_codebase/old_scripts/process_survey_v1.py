# First version of survey processing
# Before we realized the encoding issues
# DEPRECATED

import pandas as pd

def process():
    df = pd.read_csv('data/raw_survey.csv')
    # This didn't work because of special characters
    df = df.dropna()
    df.to_csv('data/survey_clean.csv')

if __name__ == '__main__':
    process()
