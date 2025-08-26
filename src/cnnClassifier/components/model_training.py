from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

from cnnClassifier.entity.config_entity import ModelTrainingConfig



STAGE_NAME = "Training"
class ModelTrainer:
    def __init__(self,config:ModelTrainingConfig):
        self.config = config

    def train(self, train_df: DataFrame):
        lr = LogisticRegression(labelCol=self.config.label_col, featuresCol=self.config.features_col)
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, self.config.reg_params) \
            .addGrid(lr.elasticNetParam, self.config.elastic_net_params) \
            .build()
        evaluator = MulticlassClassificationEvaluator(labelCol=self.config.label_col, predictionCol="prediction", metricName="accuracy")
        cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=self.config.num_folds)
        cvModel = cv.fit(train_df)
        return cvModel
