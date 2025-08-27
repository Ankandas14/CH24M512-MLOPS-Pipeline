import mlflow
from pyspark.sql import SparkSession
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import ModelTrainer
from cnnClassifier.components.model_evaluation_mlflow import ModelEvaluator
from cnnClassifier import logger
from urllib.parse import urlparse
import os



STAGE_NAME = "Training"

class ModelTrainingPipeline:
	def __init__(self):
		self.config_manager = ConfigurationManager()
		self.model_config = self.config_manager.get_model_training_config()
		self.spark = self.get_spark()

	def get_spark(self):
		return SparkSession.builder \
			.appName("Titanic-Model-Training") \
			.config("spark.driver.memory", "4g") \
			.config("spark.executor.memory", "4g") \
			.getOrCreate()

	def main(self):
		import time
		mlflow.set_tracking_uri("file:///mlruns")
		
		mlflow.set_experiment("Titanic_Distributed_Training")
		with mlflow.start_run(run_name="LogisticRegression") as run:
			# Load data
			train_df = self.spark.read.parquet(self.model_config.train_data_path)
			test_df = self.spark.read.parquet(self.model_config.test_data_path)
			# Train model
			logger.info("Training started.")
			start_time = time.time()
			trainer = ModelTrainer(self.model_config)
			cvModel = trainer.train(train_df)
			end_time = time.time()
			logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
			# Log best model parameters
			best_reg_param = cvModel.bestModel._java_obj.getRegParam()
			best_enet_param = cvModel.bestModel._java_obj.getElasticNetParam()
			logger.info(f"Best model parameters: regParam={best_reg_param}, elasticNetParam={best_enet_param}")
			mlflow.log_param("best_regParam", best_reg_param)
			mlflow.log_param("best_elasticNetParam", best_enet_param)
			# Evaluate and log
			logger.info("Testing started.")
			evaluator = ModelEvaluator(self.model_config)
			accuracy = evaluator.evaluate_and_log(cvModel, test_df,mlflow)
			# Log model to MLflow
			model_info = mlflow.spark.log_model(cvModel.bestModel, "model")
			logger.info(f"Best accuracy: {accuracy}")
			logger.info(f"Model saved. Full Model Info: {model_info}")
			logger.info(f"Model URI (for loading): {model_info.model_uri}")
			logger.info(f"Full Artifact Path (on disk): {os.path.join(run.info.artifact_uri, model_info.artifact_path)}")
			# MLflow Model Registry: Register and transition model
			model_version = evaluator.model_evaluation_mlflow(run, model_name="Titanic-Model")
			logger.info(f"Model Registry: Model version {model_version} successfully registered and transitioned.")
			# Copy best model artifact for DVC tracking
			experiment_id = run.info.experiment_id
			best_run_id = run.info.run_id
			artifact_uri = run.info.artifact_uri
			logger.info(f"Experiment ID: {experiment_id}, Best Run ID: {best_run_id}")
			logger.info(f"The path is : {artifact_uri}")
			best_model_path = f"{artifact_uri}/model"
			evaluator.copy_best_model(best_model_path, "artifacts/best_model/")
		self.spark.stop()






if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e