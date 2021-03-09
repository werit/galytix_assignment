from gensim.models import KeyedVectors


class FlatFileCreationService:
    def make_flat_file(self, location, target_vector_file_name):
        wv = KeyedVectors.load_word2vec_format(
            location, binary=True, limit=1000000)
        wv.save_word2vec_format(target_vector_file_name)
