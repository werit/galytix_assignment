from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType


class DistanceCalculationService:

    def calculate_distance(self, input_df):
        squared_distance = udf(
            lambda vect1, vect2: float(Vectors.squared_distance(vect1, vect2)), FloatType())
        ad = input_df.alias('df1').join(
            input_df.alias('df2'), col('df1.sentence_id') != col('df2.sentence_id'), 'inner')
        ad2 = ad.select(col('df1.sentence_id').alias('sentence_id'),
                        col('df2.sentence_id').alias('sentence_id_match'),
                        col('df1.sentence_vector').alias('sentence_vector'),
                        col('df2.sentence_vector').alias(
                            'sentence_vector_match')
                        )
        return ad2.withColumn('distance', squared_distance(
            col('sentence_vector'), col('sentence_vector_match')))
