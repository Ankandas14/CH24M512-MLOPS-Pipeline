import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, mean, udf
from pyspark.sql.types import IntegerType, DoubleType, StringType
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import TitanicPreprocessingConfig
from pathlib import Path
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

class TitanicPreprocessing:
    def assemble_features(self, df):
        """
        Assembles all columns except 'Survived' and 'features' into a single feature vector column called 'features'.
        Retains the 'Survived' column in the output DataFrame.
        """
        from pyspark.ml.feature import VectorAssembler
        input_cols = [col for col in df.columns if col not in ["Survived", "features"]]
        assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
        df = assembler.transform(df)
        # Ensure 'Survived' is present
        if "Survived" in df.columns:
            df = df.select(["Survived"] + [c for c in df.columns if c != "Survived"])
        return df
    def vector_assemble_features(self, df, input_cols, output_col="features"):
        """
        Transforms the DataFrame by assembling the specified input columns into a single vector column using PySpark's VectorAssembler.
        Args:
            df: Input Spark DataFrame
            input_cols: List of column names to assemble
            output_col: Name of the output vector column (default: 'features')
        Returns:
            DataFrame with the new vector column
        """
        from pyspark.ml.feature import VectorAssembler
        assembler = VectorAssembler(inputCols=input_cols, outputCol=output_col)
        df = assembler.transform(df)
        return df
    def __init__(self, config: TitanicPreprocessingConfig):
        self.config = config
        self.spark = None
        self.processed_root = Path(config.processed_root)

    def build_spark(self, app_name="Titanic-Spark-ETL"):
        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.hadoop.io.native.lib.available", "false")
            .config("spark.sql.parquet.output.committer.class", "org.apache.parquet.hadoop.ParquetOutputCommitter")
            .config("spark.driver.memory", "6g")
            .config("spark.executor.memory", "6g")
            .getOrCreate()
)
        logger.info(f"Spark session started with app name: {app_name}")

    def load_data(self, csv_path):
        return self.spark.read.csv(csv_path, header=True, inferSchema=True)

    def preprocess(self, df):
        # Fill missing Age with mean
        mean_age = df.select(mean(col("Age"))).collect()[0][0]
        df = df.withColumn("Age", when(col("Age").isNull(), mean_age).otherwise(col("Age")))
        # Fill missing Embarked with most common
        most_common_embarked = df.groupBy("Embarked").count().orderBy(col("count").desc()).first()[0]
        df = df.withColumn("Embarked", when(col("Embarked").isNull(), most_common_embarked).otherwise(col("Embarked")))
        # Fill missing Fare with mean
        mean_fare = df.select(mean(col("Fare"))).collect()[0][0]
        df = df.withColumn("Fare", when(col("Fare").isNull(), mean_fare).otherwise(col("Fare")))
        # Convert categorical columns to numerical using StringIndexer and Pipeline
        from pyspark.ml.feature import StringIndexer
        from pyspark.ml import Pipeline
        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in ["Sex","Embarked"]]
        pipeline = Pipeline(stages=indexers)
        df = pipeline.fit(df).transform(df)
        # Scale Age and Fare using a separate method
        df = self.scale_numerical_features(df)
        # Drop columns not needed
        df = df.drop("Name", "Ticket", "Cabin", "PassengerId", "Sex", "Embarked")
        # Assemble all features except the first column
        df = self.assemble_features(df)
        return df
    def scale_numerical_features(self, df):
        """
        Scales the 'Age' and 'Fare' columns using PySpark's StandardScaler and adds new columns 'Age_scaled' and 'Fare_scaled'.
        """
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        from pyspark.sql.functions import udf as spark_udf
        from pyspark.sql.types import DoubleType
        # Assemble features
        assembler = VectorAssembler(inputCols=["Age", "Fare"], outputCol="features_to_scale")
        df = assembler.transform(df)
        # StandardScaler
        scaler = StandardScaler(inputCol="features_to_scale", outputCol="scaled_features", withMean=True, withStd=True)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        # Extract scaled Age and Fare
        def extract_age(features):
            return float(features[0]) if features else None
        def extract_fare(features):
            return float(features[1]) if features else None
        extract_age_udf = spark_udf(extract_age, DoubleType())
        extract_fare_udf = spark_udf(extract_fare, DoubleType())
        df = df.withColumn("Age_scaled", extract_age_udf(col("scaled_features")))
        df = df.withColumn("Fare_scaled", extract_fare_udf(col("scaled_features")))
        df = df.drop("features_to_scale", "scaled_features")
        return df

    def write_data(self, df, out_path):
        df.write.mode("overwrite").parquet(out_path)
        logger.info(f"Preprocessed Titanic data written to {out_path}")
    def view_parquet(self, parquet_path):
        df = self.spark.read.parquet(parquet_path)
        shape = (df.count(), len(df.columns))
        logger.info(f"Shape of {parquet_path}: {shape}")
        logger.info(f"First 5 rows of {parquet_path}:")
        df.show(5)
        logger.info(f"Summary of {parquet_path}:")
        df.describe().show()

    def run(self,):
        self.build_spark()
        train_path = self.config.train_csv
        # Load full train.csv
        full_df = self.load_data(train_path)
        # Split into train/test (80/20)
        train_df, test_df = full_df.randomSplit([0.8, 0.2], seed=42)
        train_df = self.preprocess(train_df)
        test_df = self.preprocess(test_df)
        train_out = str(self.processed_root / "train.parquet")
        test_out = str(self.processed_root / "test.parquet")
        self.write_data(train_df, train_out)
        self.write_data(test_df, test_out)
        self.view_parquet(train_out)
        self.view_parquet(test_out)
        self.spark.stop()
        logger.info(f"Spark session stopped. Output dir: {self.processed_root}")
        return str(self.processed_root)
