# Download survey responses from Qualtrics
# Our primary data source
# Created: Jan 2022

import requests
import pandas as pd
import zipfile
import io

QUALTRICS_API_TOKEN = "your_token"
DATACENTER = "ca1"  # Check your Qualtrics account
SURVEY_ID = "SV_xxxxxxxxxxxxx"

def download_survey_responses():
    """
    Download all responses from our main survey.
    This is the primary data for chapters 3-4.
    """

    base_url = f"https://{DATACENTER}.qualtrics.com/API/v3/surveys/{SURVEY_ID}/export-responses"

    headers = {
        "X-API-TOKEN": QUALTRICS_API_TOKEN,
        "Content-Type": "application/json"
    }

    # Start export
    data = {"format": "csv"}
    response = requests.post(base_url, json=data, headers=headers)
    progress_id = response.json()["result"]["progressId"]

    # Check progress (simplified - real code would poll)
    progress_url = f"{base_url}/{progress_id}"

    # ... polling logic here ...

    # Download file
    file_id = "xxx"  # from progress response
    download_url = f"{base_url}/{file_id}/file"
    download_response = requests.get(download_url, headers=headers)

    # Extract CSV from zip
    zipfile_obj = zipfile.ZipFile(io.BytesIO(download_response.content))
    csv_name = zipfile_obj.namelist()[0]
    df = pd.read_csv(zipfile_obj.open(csv_name))

    # Save raw
    df.to_csv('C:/Users/PhDStudent/Desktop/thesis_final/data_raw/survey_responses_raw.csv', index=False)
    print(f"Downloaded {len(df)} survey responses")

    return df

def download_wave2():
    """Second wave of survey - added Oct 2023"""
    # Same as above but different survey ID
    SURVEY_ID_W2 = "SV_yyyyyyyyyyy"
    # ... similar code ...
    pass

if __name__ == '__main__':
    download_survey_responses()
    # download_wave2()  # uncomment for wave 2
