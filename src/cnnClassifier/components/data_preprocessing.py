
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataPreprocessingConfig
from pathlib import Path
import idx2numpy
import numpy as np
from PIL import Image
import io
import uuid
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.spark = None
        self.resize_w, self.resize_h = config.resize
        self.processed_root = Path(config.processed_root)

    def build_spark(self, app_name="MNIST-Spark-ETL"):
        self.spark = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .config("spark.driver.host", "127.0.0.1") \
            .getOrCreate()
        )
        logger.info(f"Spark session started with app name: {app_name}")

    def _load_idx(self, img_path: str, lbl_path: str):
        X = idx2numpy.convert_from_file(img_path)
        y = idx2numpy.convert_from_file(lbl_path)
        X = X.reshape((X.shape[0], -1))
        rows = [(int(y[i]), X[i].astype(int).tolist()) for i in range(X.shape[0])]
        from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
        schema = StructType([
            StructField("label", IntegerType(), False),
            StructField("pixels", ArrayType(IntegerType(), False), False)
        ])
        return self.spark.createDataFrame(rows, schema)


    def load_train(self):
        d = self.config.mnist
        return self._load_idx(d["train_images"], d["train_labels"]).withColumn("split", lit("train"))

    def load_test(self):
        d = self.config.mnist
        return self._load_idx(d["test_images"], d["test_labels"]).withColumn("split", lit("test"))

    @staticmethod
    def _to_png_bytes(pixels, label, resize_w, resize_h):
        arr = np.array(pixels, dtype=np.uint8).reshape(28, 28)
        p_min, p_max = int(arr.min()), int(arr.max())
        if p_max > p_min:
            arr = ((arr - p_min) * (255.0 / (p_max - p_min))).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").resize((resize_w, resize_h), resample=Image.BILINEAR).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        # Return as list of ints for Spark compatibility
        return list(buf.getvalue())

    def transform_and_write(self, df):
        resize_w, resize_h = self.resize_w, self.resize_h
        from pyspark.sql.functions import udf, col, monotonically_increasing_id
        from pyspark.sql.types import ArrayType, IntegerType
        to_png_udf = udf(
            lambda p, y: DataPreprocessing._to_png_bytes(p, y, resize_w, resize_h),
            returnType=ArrayType(IntegerType(), False)
        )
        df_bytes = df.withColumn("png_bytes", to_png_udf(col("pixels"), col("label"))) \
                     .withColumn("id", monotonically_increasing_id())
        out_root = str(self.processed_root.resolve())
        def _write_partition(rows):
            for r in rows:
                split = r["split"]
                label = r["label"]
                try:
                    raw_bytes = r["png_bytes"]
                    invalid = [x for x in raw_bytes if not (0 <= int(x) <= 255)]
                    if invalid:
                        logger.error(f"Out-of-range bytes for label {label}, split {split}: {invalid}")
                        continue
                    bts = bytes([int(x) for x in raw_bytes])
                    fname = f"{uuid.uuid4().hex}.png"
                    out_dir = Path(out_root) / split / str(label)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    with open(out_dir / fname, "wb") as f:
                        f.write(bts)
                except Exception as e:
                    logger.error(f"Exception in writing PNG for label {label}, split {split}: {e}")
                    continue
        df_bytes.select("split", "label", "png_bytes").foreachPartition(_write_partition)
        logger.info(f"Preprocessed images written to {out_root}")
        return str(self.processed_root)

    def run(self):
        self.build_spark()
        train_df = self.load_train()
        test_df = self.load_test()
        full_df = train_df.unionByName(test_df)
        out_dir = self.transform_and_write(full_df)
        self.spark.stop()
        logger.info(f"Spark session stopped. Output dir: {out_dir}")
        return out_dir
