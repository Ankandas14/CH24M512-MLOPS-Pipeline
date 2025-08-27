import mlflow
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_02_spark_data_pipeline import DataPreprocessingPipeline
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.prediction import PredictionPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   #data_ingestion = DataIngestionTrainingPipeline()
  # data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Spark Data Preprocessing stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   # preprocessing = DataPreprocessingPipeline()
    #preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #training = ModelTrainingPipeline()
    #training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prediction stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    mlflow.set_tracking_uri("file:///C:/mlruns")
     # Example data point based on Titanic dataset features
    sample_data = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    predictionPipeline = PredictionPipeline()
    predictionPipeline.predict(sample_data)
    #training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e




