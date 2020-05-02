# https://app.pluralsight.com/guides/handling-structured-data-in-pyspark-with-dataframes

from pyspark.sql import SparkSession

# all communication with the cluster is performed through a SparkSession object.
# Therefore, our first step is to create one.
spark = SparkSession.builder.getOrCreate()

# In Spark, data is represented by DataFrame objects,
# which can be thought of as a 2D structure following the tidy data format.
# This means that each row represents an observation and each column a variable; accordingly, columns must have names and types.
# Creating from Python object
df = spark.createDataFrame([['Japan', 'Asia', 126.8],
							['Portugal', 'Europe',10.31],
							['Germany', 'Europe', 82.79],
							['China', 'Asia', 1386.0],
							['Pakistan', 'Asia', 197.0],
							['Brazil', 'South America', 209.3],
							['Spain', 'Europe', 46.72]],
						   ['name', 'continent', 'population'])

df.show()

# Reading from a Parquet archive stored at path/to/my_parquet_data

parquet_df = spark.read.parquet('path/to/my_parquet_data')

# Reading from a Hive table mytable stored in the database mydatabase

spark.sql('use mydatabase')
hive_df = spark.read.table('mytable')

# One of the things that make Spark efficient is
# lazy evaluation: it defers calculations until their
# results are actually needed.

# they are called actions.

# The collect method, which transfers data from the executors to the driver
# The count method, which counts the number of rows in the DataFrame

# Example:
df.groupby().mean('population').show()

# To conclude, note that show is something like a print statement.
# To use values that, just replace show above with collect, which will return a list of Row objects.