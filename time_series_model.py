import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

# Load the data
file_path = 'btc_usd_hourly.csv'
data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
data.sort_index(inplace=True)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data['priceUsd'][:train_size], data['priceUsd'][train_size:]

# Define and fit the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Calculate metrics
mse = mean_squared_error(test, predictions)
rmse = mean_squared_error(test, predictions, squared=False)
mae = mean_absolute_error(test, predictions)
r2 = r2_score(test, predictions)

# Log the experiment with MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("bitcoin_price_prediction_ARIMA")

# Check for active run and end it if necessary
if mlflow.active_run() is not None:
    mlflow.end_run()

try:
    with mlflow.start_run():
        mlflow.log_param("order", (5, 1, 2))
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_text(model_fit.summary().as_text(), "model_summary.txt")

        print('Metrics and model summary logged to MLflow.')
finally:
    mlflow.end_run()