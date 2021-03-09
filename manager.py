import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max

from services.distance_service import DistanceCalculationService
from services.flat_file_service import FlatFileCreationService
from services.spark_df_read_service import SparkDfReadService
from services.word_embedding_service import WordEmbeddingService


class AssignmentManager:

    def __init__(self):
        self.sentence_col_name = 'sentence'
        self.sentence_col_id = 'sentence_id'
        self.word_col_name = 'word'
        self.vector_csv = 'vectors.csv'
        self.phrases_csv = 'phrases.csv'
        self.docker_spark_home_dir = '/home/jovyan/work/'
        self.google_word_vector_bin = 'GoogleNews-vectors-negative300.bin'
        self.ss = SparkSession.builder.appName('assignment').getOrCreate()
        self.flat_file_service = FlatFileCreationService()
        self.dist_service = DistanceCalculationService()
        self.word_embedding_service = WordEmbeddingService(self.ss)
        self.read_service = SparkDfReadService(self.ss)

    def manage(self, sentence: str):
        self.make_flat_file()

        source_df = self.read_service.read_spark_csv_with_row_id(self.docker_spark_home_dir + self.phrases_csv, '//')
        vector_df = self.read_service.read_word_vector(self.docker_spark_home_dir + self.vector_csv, ' ')

        if sentence is None:
            self.get_general_distances(source_df, vector_df)
        else:
            self.get_closest_sentence(sentence, source_df, vector_df)

    def make_flat_file(self):
        # TODO: should use try catch if path does not exist
        if not os.path.isfile(self.docker_spark_home_dir + self.vector_csv):
            self.flat_file_service.make_flat_file(
                f'file://{self.docker_spark_home_dir}{self.google_word_vector_bin}', self.vector_csv)

    def get_general_distances(self, source_df, vector_df):
        return self.calculate_word_distances(source_df, vector_df)

    def get_closest_sentence(self, sentence, source_df, vector_df):
        current_max_row = int(
            source_df.agg(max(self.sentence_col_id).alias('max_id')).select('max_id').collect()[0]['max_id'])

        source_df = self.add_sentence_to_df(current_max_row, sentence, source_df)
        source_df.persist()

        dist = self.calculate_word_distances(source_df, vector_df)

        dist.where(col(self.sentence_col_id) == current_max_row + 1) \
            .orderBy(col('distance')).take(2).toDf().alias('df1').join(source_df.alais('df2'),
                                                                       col('df1.sentence_id_match') == col(
                                                                           'df2.sentence_id')).show()

    def calculate_word_distances(self, source_df, vector_df):
        vector_df.persist()
        df = self.word_embedding_service.embed_words(source_df, vector_df)
        df.persist()
        return self.dist_service.calculate_distance(df)

    def add_sentence_to_df(self, max_row, sentence, source_df):
        sentence_df = self.ss.createDataFrame([(sentence, max_row + 1)], [self.sentence_col_name, self.sentence_col_id])
        return source_df.union(sentence_df)
