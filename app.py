from flask import Flask, jsonify
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import logging
import os

app = Flask(__name__)

DATA_URL = 'https://api.coincap.io/v2/assets/bitcoin/history'
MODEL_FILE = 'random_forest_model.pkl'
SCALER_FILE = 'scaler.pkl'

logging.basicConfig(level=logging.INFO)

def fetch_data():
    try:
        END_DATE = datetime.utcnow()
        START_DATE = END_DATE - timedelta(days=15)
        response = requests.get(DATA_URL, params={
            'interval': 'h1',
            'start': int(START_DATE.timestamp() * 1000),
            'end': int(END_DATE.timestamp() * 1000)
        })
        response.raise_for_status()
        data = response.json().get('data', [])
        if not data:
            logging.error("No data received from API.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['time']/1000, unit='s')
        df.set_index('date', inplace=True)
        df['priceUsd'] = df['priceUsd'].astype(float)
        df['circulatingSupply'] = df['circulatingSupply'].astype(float)
        return df[['priceUsd', 'circulatingSupply']]
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def preprocess_data(data):
    try:
        data['priceUsd_lag1'] = data['priceUsd'].shift(1)
        data['priceUsd_lag2'] = data['priceUsd'].shift(2)
        data['priceUsd_lag3'] = data['priceUsd'].shift(3)
        data.dropna(inplace=True)
        X = data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
        return X
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

@app.route('/predict', methods=['GET'])
def predict():
    data = fetch_data()
    if data.empty:
        return jsonify({'error': 'No data available'}), 500
    
    X = preprocess_data(data)
    if X.empty:
        return jsonify({'error': 'Data preprocessing failed'}), 500
    
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled[-1].reshape(1, -1))
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logging.info(f"Starting Flask app on port {port}")
    print(f"Starting Flask app on port {port}")  # Additional print statement for debugging
    app.run(debug=True, host='0.0.0.0', port=port)