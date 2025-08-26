
import mlflow
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from cnnClassifier import logger
import shutil
import os
import os
import shutil
from urllib.parse import urlparse

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

    def evaluate_and_log(self, model, test_df: DataFrame,mlflow):
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
        logger.info(f"evaluate_and_log: {accuracy}")
        return accuracy
    
    # Make sure these are imported at the top of your file


    def copy_best_model(self, best_model_path, target_dir="artifacts/best_model/"):
        # --- START: ADDED CODE TO HANDLE URI ---
        # Parse the path to check if it's a file URI
        parsed_path = urlparse(best_model_path)
        
        # If the scheme is 'file', convert it to a standard local path
        if parsed_path.scheme == 'file':
            # The .path attribute holds the local path part of the URI
            local_model_path = parsed_path.path
            # On Windows, urlparse leaves a leading '/' (e.g., /C:/...), so remove it
            if os.name == 'nt' and local_model_path.startswith('/'):
                local_model_path = local_model_path[1:]
        else:
            # If it's not a file URI, assume it's already a valid local path
            local_model_path = best_model_path
        # --- END: ADDED CODE ---

        os.makedirs(target_dir, exist_ok=True)
        
        # IMPORTANT: Use the newly cleaned 'local_model_path' from now on
        if os.path.isdir(local_model_path):
            # Use shutil.copytree for a much simpler and safer directory copy
            for item in os.listdir(local_model_path):
                s = os.path.join(local_model_path, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
        else:
            shutil.copy2(local_model_path, target_dir)