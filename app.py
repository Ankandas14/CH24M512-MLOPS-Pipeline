from flask import Flask, request, jsonify
import os
import pandas as pd
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger

# --- MLflow Configuration ---
# This script requires the MLFLOW_TRACKING_URI environment variable to be set.
# It tells the app where to find your mlruns directory.
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not mlflow_uri:
    logger.error("FATAL: MLFLOW_TRACKING_URI environment variable is not set.")
    logger.error("Please set it to the absolute path of your mlruns directory.")
    logger.error("Example (Windows): set MLFLOW_TRACKING_URI=file:///C:/mlruns")
    logger.error("Example (Linux/WSL): export MLFLOW_TRACKING_URI='file:///home/user/project/mlruns'")
    exit(1)

logger.info(f"Using MLflow tracking URI: {mlflow_uri}")

app = Flask(__name__)

# --- Initialize Prediction Pipeline ---
# The pipeline is created once when the app starts to avoid reloading the model on every request.
try:
    # Ensure the model name here matches the one you registered in MLflow
    prediction_pipeline = PredictionPipeline(model_name="Titanic-Model", stage="Production")
except Exception as e:
    logger.error(f"Failed to initialize the Prediction Pipeline during startup: {e}")
    prediction_pipeline = None

@app.route('/', methods=['GET'])
def home():
    """A simple health check endpoint to confirm the API is running."""
    return "Titanic Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    The main prediction endpoint.
    Accepts JSON data with features and returns a prediction.
    """
    if prediction_pipeline is None:
        return jsonify({"error": "Prediction service is unavailable. Check server logs for initialization errors."}), 500

    try:
        # Get data from the POST request
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid input: No JSON data received."}), 400

        logger.info(f"Received data for prediction: {data}")
        
        # Use the pipeline to get a prediction
        result = prediction_pipeline.predict(data)

        if result is None:
            return jsonify({"error": "Prediction failed. Check server logs for details."}), 500

        # The prediction is typically a list with one value, e.g., [1.0]
        prediction_value = result[0] if result else "N/A"

        return jsonify({"prediction": prediction_value})

    except Exception as e:
        logger.error(f"An error occurred in the /predict route: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app
    # Use debug=False in a real production environment
    app.run(host='0.0.0.0', port=8080, debug=True)
