# https://app.pluralsight.com/guides/handling-structured-data-in-pyspark-with-dataframes

from pyspark.sql import SparkSession

# all communication with the cluster is performed through a SparkSession object.
# Therefore, our first step is to create one.
spark = SparkSession.builder.getOrCreate()

# In Spark, data is represented by DataFrame objects,
# which can be thought of as a 2D structure following the tidy data format.
# This means that each row represents an observation and each column a variable; accordingly, columns must have names and types.
df = spark.createDataFrame([['Japan', 'Asia', 126.8],
							['Portugal', 'Europe',10.31],
							['Germany', 'Europe', 82.79],
							['China', 'Asia', 1386.0],
							['Pakistan', 'Asia', 197.0],
							['Brazil', 'South America', 209.3],
							['Spain', 'Europe', 46.72]],
						   ['name', 'continent', 'population'])

df.show()

