from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.stat import Summarizer
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, explode, row_number, levenshtein, count, min
from pyspark.sql.window import *


class WordEmbeddingService:
    def __init__(self, spark_session: SparkSession):
        super().__init__()
        self.spark_session = spark_session
        self.sentence_col_name = 'sentence'
        self.sentence_col_id = 'sentence_id'
        self.word_col_name = 'word'

    def tokenize(self, df: DataFrame, col_to_tokenize: str):
        regex_tokenizer = RegexTokenizer(
            inputCol=col_to_tokenize, outputCol='word', pattern='\\W')
        return regex_tokenizer.transform(df)

    def remove_stop_words(self, df: DataFrame):
        remover = StopWordsRemover(inputCol='word', outputCol='filtered')
        return remover.transform(df)

    def embed_vector_to_not_matched_words(self, df: DataFrame, df_vector_filler: DataFrame):
        not_matched_df = df.where(
            col('word_vector').isNull()).select(self.sentence_col_id, 'word')

        df3 = self.assign_alternative_match_word_based_on_lavenshtein(
            not_matched_df, df_vector_filler)

        return df3.alias('base').join(df_vector_filler.alias('filler'),
                                      df3.match == col('filler' + '.' + self.word_col_name), how='left').select(
            self.sentence_col_id, col('base' + '.' + 'word').alias('word'),
            col('filler' + '.' + 'word_vector').alias('word_vector'))

    def assign_alternative_match_word_based_on_lavenshtein(self, not_matched_df, df_vector_filler):
        not_matched_df_x_filler = not_matched_df.crossJoin(
            df_vector_filler.select(col(self.word_col_name).alias('match')))
        df1_x_df2 = not_matched_df_x_filler.withColumn("levenshtein", levenshtein(
            col('word'), col('match')))
        return df1_x_df2.withColumn('overall_min', min(col("levenshtein")).over(
            Window.partitionBy(self.sentence_col_id, 'word'))) \
            .where(col('overall_min') == col('levenshtein')) \
            .withColumn('rank', row_number().over(Window.partitionBy(self.sentence_col_id, 'word').orderBy('match'))) \
            .where(col('rank') == 1) \
            .drop('levenshtein', 'overall_min', 'rank')

    def embed_words(self, sentence_df: DataFrame, word_vector_df: DataFrame) -> DataFrame:
        tokenized_source = self.tokenize(sentence_df, self.sentence_col_name)
        tokenized_source_no_stop_words = self.remove_stop_words(
            tokenized_source)

        clean_word_no_dup_df = self.convert_tokens_to_words_no_duplicates(
            tokenized_source_no_stop_words)

        return self.match_word_with_word_vector(clean_word_no_dup_df, word_vector_df)

    def match_word_with_word_vector(self, clean_word_no_dup_df, word_vector_df):
        words_with_vector_df = self.assign_vector_to_words(
            clean_word_no_dup_df, word_vector_df)

        words_with_vector_df.persist()

        mismatched_words_matched_df = self.embed_vector_to_not_matched_words(
            words_with_vector_df, word_vector_df)

        complete_match_df = words_with_vector_df.where(
            col('word_vector').isNotNull()).union(mismatched_words_matched_df)

        return complete_match_df.groupBy('sentence_id').agg(
            Summarizer.sum(col('word_vector')).alias('sentence_vector')).select('sentence_id', 'sentence_vector')

    def assign_vector_to_words(self, clean_word_no_dup_df, word_vector_df):
        return clean_word_no_dup_df.alias('src').join(
            word_vector_df.alias('vect'), col(
                'src' + '.' + 'word') == col('vect' + '.' + self.word_col_name),
            how='left') \
            .select(self.sentence_col_id, col('src' + '.' + 'word').alias('word'), 'word_vector')

    def convert_tokens_to_words_no_duplicates(self, tokenized_source_no_stop_words):
        clean_word_df = tokenized_source_no_stop_words \
            .drop('word') \
            .withColumn("word", explode('filtered')).alias("word") \
            .select(self.sentence_col_id, "word")
        # removing duplicates
        clean_word_no_dup_df = clean_word_df \
            .groupBy(self.sentence_col_id, "word") \
            .agg(count('word').alias('cnt')) \
            .select(self.sentence_col_id, "word")
        return clean_word_no_dup_df
