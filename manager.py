from services.flat_file_service import FlatFileCreationService
from services.word_embedding_service import WordEmbedder
import os
from pyspark.sql import SparkSession


class AssignmentManager:

    def make_flat_file(self):
        # TODO: should use try catch if path does not exist
        if not os.path.isfile('/home/jovyan/work/vectors.csv'):
            flat_file_service = FlatFileCreationService()
            flat_file_service.make_flat_file(
                'file:///home/jovyan/work/GoogleNews-vectors-negative300.bin')

    def assign_word_embedding(self, ss: SparkSession):
        word_embedder = WordEmbedder(ss)
        word_embedder.embed_words()

    def manage(self):
        self.make_flat_file()
        ss = SparkSession.builder.appName('assignment').getOrCreate()
        self.assign_word_embedding(ss)
