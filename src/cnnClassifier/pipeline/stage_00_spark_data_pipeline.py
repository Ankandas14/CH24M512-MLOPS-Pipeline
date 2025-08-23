from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing import DataPreprocessing
from cnnClassifier import logger

STAGE_NAME = "Spark Data Preprocessing stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessing_config = config.get_data_preprocessing_config()
        preprocessing = DataPreprocessing(preprocessing_config)
        preprocessing.run()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
