import pyspark
from pyspark.sql import SparkSession
import pandas as pd

hotel_ratings = pd.read_csv('dataset/hotel_ratings.csv')

# Setup a SparkSession
spark = SparkSession.builder.getOrCreate()

# Convert a Pandas DF to a Spark DF
spark_hotel_ratings = spark.createDataFrame(hotel_ratings)

# Convert a Spark DF to a Pandas DF
pandas_df = spark_df.toPandas()
