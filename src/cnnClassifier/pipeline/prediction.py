import mlflow
import pandas as pd
from pyspark.sql import SparkSession

class ModelPredictor:
    def __init__(self, model_name="Titanic-Model", stage="Production"):
        self.model_name = model_name
        self.stage = stage
        self.spark = SparkSession.builder.appName("Titanic-Prediction").getOrCreate()
        self.model = self.load_model()

    def load_model(self):
        model_uri = f"models:/{self.model_name}/{self.stage}"
        return mlflow.spark.load_model(model_uri)

    def predict(self, input_df: pd.DataFrame):
        spark_df = self.spark.createDataFrame(input_df)
        predictions = self.model.transform(spark_df)
        return predictions.select("prediction").toPandas()["prediction"].tolist()

    def stop(self):
        self.spark.stop()

# Example usage:
# predictor = ModelPredictor()
# result = predictor.predict(pd.DataFrame({...}))
# predictor.stop()
