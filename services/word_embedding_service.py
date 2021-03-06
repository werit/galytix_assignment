from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.dataframe import DataFrame


class WordEmbedder:
    def __init__(self, spark_session: SparkSession):
        super().__init__()
        self.spark_session = spark_session

    def read_csv_without_delimitation(self, path_to_csv):
        return self.spark_session.read.format('csv').option("delimiter", "//").option(
            "header", "false").load(path_to_csv)

    def tokenize(self, df: DataFrame):
        regex_tokenizer = RegexTokenizer(
            inputCol='_c0', outputCol='word', pattern='\\W')
        return regex_tokenizer.transform(df)

    def remove_stop_words(self, df: DataFrame):
        remover = StopWordsRemover(inputCol='word', outputCol='filtered')
        count_tokens = udf(lambda word: len(word), IntegerType())
        return remover.transform(df).withColumn(
            'tokens', count_tokens(col('filtered')))

    def embed_words(self):
        # do not split by delimitation
        df_source = self.read_csv_without_delimitation(
            '/home/jovyan/work/phrases.csv')
        df_vect = self.spark_session.read.format('csv').option("delimiter", " ").option(
            "header", "false").load('/home/jovyan/work/vectors.csv')

        tokenized_source = self.tokenize(df_source)
        tokenized_source_no_stop_words = self.remove_stop_words(
            tokenized_source)
        print(tokenized_source_no_stop_words.take(20))
        print(df_vect.select(df_vect.columns[:2]).take(5))
