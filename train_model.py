import mlflow
import mlflow.sklearn

# Start MLflow experiment
mlflow.set_experiment('btc_price_prediction')  # Replace with your experiment name
mlflow.start_run(run_name='initial_run')  # Start a new run
