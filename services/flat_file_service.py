from gensim.models import KeyedVectors


class FlatFileCreationService:
    def make_flat_file(self, location):
        wv = KeyedVectors.load_word2vec_format(
            location, binary=True, limit=1000000)
        wv.save_word2vec_format('vectors.csv')
