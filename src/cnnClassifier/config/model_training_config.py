from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    train_data_path: str
    test_data_path: str
    label_col: str = "Survived"
    features_col: str = "features"
    reg_params: list = None
    elastic_net_params: list = None
    num_folds: int = 3

    def __post_init__(self):
        if self.reg_params is None:
            self.reg_params = [0.01, 0.1, 1.0]
        if self.elastic_net_params is None:
            self.elastic_net_params = [0.0, 0.5, 1.0]
