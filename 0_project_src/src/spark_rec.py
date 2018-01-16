import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyspark
from pyspark.sql.types import *
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
# https://www.codementor.io/jadianes/building-a-recommender-with-apache-spark-python-example-app-part1-du1083qbw
# https://github.com/dipanjanS/BerkeleyX-CS100.1x-Big-Data-with-Apache-Spark/blob/master/Week%205%20-%20Introduction%20to%20Machine%20Learning%20with%20Apache%20Spark/lab4_machine_learning_student.ipynb

spark = pyspark.sql.SparkSession.builder.getOrCreate()
sc = spark.sparkContext

from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('dataset/unique_popular_3k_hotels.csv')
#df.select('year', 'model').write.format('com.databricks.spark.csv').save('newcars.csv')



from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in list(set(df.columns)) ]

pipeline = Pipeline(stages=indexers)
ratings = pipeline.fit(df).transform(df)



train, validation, test = ratings.randomSplit([0.6, 0.2, 0.2], seed=427471138)

als_model = ALS(userCol='user_index',
                itemCol='hotel id',
                ratingCol='ratings',
                nonnegative=True,
                regParam=0.1,
                rank=10
               )

recommender = als_model.fit(train)


# Build a single row DataFrame
data = [(1, 100)]
columns = ('user', 'movie')
one_row_spark_df = spark.createDataFrame(data, columns)

user_factor_df = recommender.userFactors.filter('id = 1')
item_factor_df = recommender.itemFactors.filter('id = 100')

user_factors = user_factor_df.collect()[0]['features']
item_factors = item_factor_df.collect()[0]['features']


# Get the recommender's prediction
recommender.transform(one_row_spark_df).show()

predictions = recommender.transform(test)
