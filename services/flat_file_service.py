from gensim.models import KeyedVectors


class FlatFileCreationService:
    def make_flat_file(self, location, target_vector_file_name):
        wv = KeyedVectors.load_word2vec_format(
            location, binary=True, limit=1000000)
        wv.save_word2vec_format(target_vector_file_name)
        # first line contains metadata
        self.remove_first_line_from_file(target_vector_file_name)

    def remove_first_line_from_file(self, file_name):
        with open(file_name, 'r+') as f:
            f.readline()
            data = f.read()
            f.seek(0)
            f.write(data)
            f.truncate()
