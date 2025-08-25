
import mlflow
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from cnnClassifier import logger

class ModelEvaluator:
    def model_evaluation_mlflow(self, run, model_name="Titanic-Model"):
        """
        Registers the best model in MLflow Model Registry and transitions it from Staging to Production.
        Returns the model version number on success.
        """
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run_id = run.info.run_id
        # Register model
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass  # Model may already exist
        model_uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(model_name, model_uri, run_id)
        logger.info(f"Model version {mv.version} registered in MLflow Model Registry.")
        # Transition model to Staging then Production
        client.transition_model_version_stage(model_name, mv.version, stage="Staging")
        logger.info(f"Model version {mv.version} transitioned to Staging.")
        client.transition_model_version_stage(model_name, mv.version, stage="Production")
        logger.info(f"Model version {mv.version} transitioned to Production.")
        return mv.version
    def __init__(self, config):
        self.config = config

    def evaluate_and_log(self, model, test_df: DataFrame):
        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(labelCol=self.config.label_col, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        mlflow.log_metric("accuracy", accuracy)
        logger.info(f"Model accuracy: {accuracy}")
        # Confusion matrix
        y_true = [row[self.config.label_col] for row in predictions.select(self.config.label_col).collect()]
        y_pred = [row["prediction"] for row in predictions.select("prediction").collect()]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,5))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        return accuracy
