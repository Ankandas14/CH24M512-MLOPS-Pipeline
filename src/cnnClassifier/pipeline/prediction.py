import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from cnnClassifier import logger
import os
import sys

# --- START: HADOOP_HOME FIX ---
# Using forward slashes is the most reliable way to handle Windows paths for Spark/Java.
hadoop_home_path = 'C:/Software/Hadoop' # Using forward slashes to avoid escape character issues.
winutils_path = os.path.join(hadoop_home_path, 'bin', 'winutils.exe')

if not os.path.exists(winutils_path):
    error_msg = (
        f"FATAL: winutils.exe not found at {winutils_path}. "
        "Please download it from a reputable source (e.g., steveloughran/winutils on GitHub) "
        "and place it in the correct directory."
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

os.environ['HADOOP_HOME'] = hadoop_home_path
os.environ['PATH'] = f"{os.path.join(hadoop_home_path, 'bin')};{os.environ['PATH']}"
# --- END: HADOOP_HOME FIX ---


class PredictionPipeline:
    """
    This class handles loading the production model from the MLflow Model Registry
    and making predictions.
    """
    def __init__(self, model_name="Titanic-Model", stage="Production"):
        """
        Initializes the pipeline by creating a Spark session and loading the model.
        
        Args:
            model_name (str): The name of the model in the MLflow Model Registry.
            stage (str): The stage of the model to load (e.g., "Production", "Staging").
        """
        self.model_name = model_name
        self.stage = stage
        self.spark = None
        self.model = None
        self.build_spark()
        self._load_model()

    def build_spark(self, app_name="Titanic-Spark-Prediction"):
        """
        Builds and configures the Spark session.
        """
        # Stop any existing session to ensure our new configuration is applied.
        active_session = SparkSession.getActiveSession()
        if active_session:
            logger.info("An active Spark session was found. Stopping it to apply new configuration.")
            active_session.stop()

        # Directly inject the hadoop.home.dir property into the Java process.
        from pyspark import SparkConf
        conf = SparkConf()
        conf.set("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={hadoop_home_path}")
        conf.set("spark.executor.extraJavaOptions", f"-Dhadoop.home.dir={hadoop_home_path}")

        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .config(conf=conf) # Apply the new configuration
            .getOrCreate()
        )

    def _load_model(self):
        """
        Loads the specified model and stage from the MLflow Model Registry.
        """
        try:
            model_uri = f"models:/{self.model_name}/{self.stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model '{self.model_name}' version '{self.stage}' from MLflow Registry.")
        except Exception as e:
            logger.error(f"Failed to load model from MLflow Registry: {e}")
            raise

    def predict(self, data):
        """
        Makes a prediction on the input data.
        """
        if self.model is None:
            logger.error("Model is not loaded. Cannot perform prediction.")
            return None

        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise TypeError("Input data must be a dictionary or Pandas DataFrame.")

            predictions = self.model.predict(df)
            logger.info(f"Prediction successful. Result: {predictions}")
            return predictions.tolist()

        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            raise
