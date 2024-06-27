import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data.sort_index(inplace=True)
    return data

# Preprocess the data
def preprocess_data(data):
    data['priceUsd_lag1'] = data['priceUsd'].shift(1)
    data['priceUsd_lag2'] = data['priceUsd'].shift(2)
    data['priceUsd_lag3'] = data['priceUsd'].shift(3)
    data.dropna(inplace=True)
    X = data[['priceUsd_lag1', 'priceUsd_lag2', 'priceUsd_lag3', 'circulatingSupply']]
    y = data['priceUsd']
    return X, y

# Train the model
def train_model(X_train, y_train, n_estimators=100, random_state=42):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Evaluate the model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)  # Transform test data using fitted scaler
    y_pred = model.predict(X_test_scaled)

    # Calculate different metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2

if __name__ == "__main__":
    # Load data
    data_file = 'btc_usd_hourly.csv'
    data = load_data(data_file)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Start MLflow run
    with mlflow.start_run():
        # Train the model and get the fitted scaler
        model, scaler = train_model(X_train, y_train)

        # Evaluate the model using the trained model and scaler
        mse, rmse, mae, r2 = evaluate_model(model, scaler, X_test, y_test)

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save the model using joblib
        joblib.dump(model, 'model.joblib')

        # Print metrics
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'R-squared (R2): {r2}')

