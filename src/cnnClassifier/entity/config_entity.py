from dataclasses import dataclass
from dataclasses import dataclass
from pathlib import Path

# For Spark-based distributed preprocessing
@dataclass(frozen=True)
class DataPreprocessingConfig:
    mnist: dict
    processed_root: str
    resize: list


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int


@dataclass(frozen=True)
class TitanicPreprocessingConfig:
    train_csv: str
    test_csv: str
    processed_root: str

@dataclass(frozen=True)
class ModelTrainingConfig:
    train_data_path: str
    test_data_path: str
    label_col: str = "Survived"
    features_col: str = "features"
    reg_params: list = None
    elastic_net_params: list = None
    num_folds: int = 3