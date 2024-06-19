import requests
import pandas as pd
import time

BASE_URL = 'https://api.coincap.io/v2/assets/bitcoin/history'

def fetch_data(interval='h1', start=None, end=None):
    """
    Fetch historical BTC/USD data on an hourly basis.

    Parameters:
    - interval: The time interval of the data (e.g., 'h1' for hourly).
    - start: The start time in milliseconds since epoch (None to fetch the most recent data).
    - end: The end time in milliseconds since epoch (None to fetch the most recent data).
    
    Returns:
    - A pandas DataFrame with the fetched data.
    """
    params = {
        'interval': interval,
    }
    if start:
        params['start'] = start
    if end:
        params['end'] = end

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if 'error' in data:
        raise Exception(f"Error fetching data: {data['error']}")

    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop(columns=['time'], inplace=True)
    return df

def fetch_all_data(start_date):
    """
    Fetch all historical BTC/USD data available in the API by making multiple requests.
    """
    all_data = []
    start_timestamp = int(pd.Timestamp.now().timestamp() * 1000)
    start_limit_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    
    while start_timestamp > start_limit_timestamp:
        df = fetch_data(start=start_limit_timestamp, end=start_timestamp)
        if df.empty:
            break
        all_data.append(df)
        start_timestamp = int(df.index.min().timestamp() * 1000) - 3600000  # Move to the previous hour
        time.sleep(1)  # To avoid hitting rate limits

    return pd.concat(all_data, ignore_index=False)

if __name__ == "__main__":
    # Set the start date for fetching data
    start_date = "2024-06-01"  # Adjust this date as needed
    data = fetch_all_data(start_date)
    data.to_csv('btc_usd_hourly.csv')
    print("Data saved to btc_usd_hourly.csv")