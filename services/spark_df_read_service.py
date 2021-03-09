from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, row_number, lit


class SparkDfReadService:
    def __init__(self, spark_session: SparkSession):
        super().__init__()
        self.spark_session = spark_session
        self.sentence_col_name = 'sentence'
        self.sentence_col_id = 'sentence_id'
        self.word_col_name = 'word'

    def read_spark_df(self, path_to_csv, delimit):
        return self.spark_session.read.format('csv').option("delimiter", delimit).option(
            "header", "false").load(path_to_csv)

    def read_word_vector(self, path_to_csv: str, delimit: str) -> DataFrame:
        vector_df = self.read_spark_csv_and_set_first_column(delimit, path_to_csv, self.word_col_name)
        vector_df = vector_df.select(self.word_col_name, *(col(c).cast("float").alias(c)
                                                           for c in vector_df.columns[1:]))
        assembler = VectorAssembler(
            inputCols=vector_df.columns[1:],
            outputCol='word_vector')
        return assembler.transform(vector_df).select(
            self.word_col_name, 'word_vector')

    def read_spark_csv_with_row_id(self, path_to_csv: str, delimit: str):
        return self.read_spark_csv_and_set_first_column(delimit, path_to_csv, self.sentence_col_name) \
            .withColumn(self.sentence_col_id, row_number().over(Window.orderBy(lit('A'))))

    def read_spark_csv_and_set_first_column(self, delimit, path_to_csv, first_column_new_name):
        vector_df = self.read_spark_df(path_to_csv, delimit)
        vector_df = vector_df.withColumnRenamed('_c0', first_column_new_name)
        return vector_df
