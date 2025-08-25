from src.cnnClassifier.config.model_training_config import ModelTrainingConfig
from pathlib import Path
import yaml
from cnnClassifier.constants import *
import os
from cnnClassifier import logger
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, DataPreprocessingConfig, TitanicPreprocessingConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
       

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
   
    
   

        return data_preprocessing_config
    
    def get_titanic_preprocessing_config(self) -> TitanicPreprocessingConfig:
        config = self.config.data_preprocessing.titanic
        return TitanicPreprocessingConfig(
            train_csv=config["train_csv"],
            test_csv=config["test_csv"],
            processed_root=config["processed_root"]
        )
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        # Support both YAML and entity-based config
        if hasattr(self, 'config') and hasattr(self.config, 'model_training'):
            config = self.config.model_training
            train_data_path = config.train_data_path
            test_data_path = config.test_data_path
            reg_params = getattr(config, 'reg_params', [0.01, 0.1, 1.0])
            elastic_net_params = getattr(config, 'elastic_net_params', [0.0, 0.5, 1.0])
            num_folds = getattr(config, 'num_folds', 3)
        else:
            # fallback to dict-style config
            config = self.config['model_training']
            train_data_path = config['train_data_path']
            test_data_path = config['test_data_path']
            reg_params = config.get('reg_params', [0.01, 0.1, 1.0])
            elastic_net_params = config.get('elastic_net_params', [0.0, 0.5, 1.0])
            num_folds = config.get('num_folds', 3)
        return ModelTrainingConfig(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            reg_params=reg_params,
            elastic_net_params=elastic_net_params,
            num_folds=num_folds
        )