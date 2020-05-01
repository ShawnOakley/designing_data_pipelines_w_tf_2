# Data preparation
# Prepping data with Pandas

# One Hot Encoding
# Process to convert a categorical column into numeric.
# Each category becomes a column with 1 or 0 representing
# whether the initial value belongs to that category

import numpy as np
import pandas as pd
import tensorflow as tf

dataset_path = tf.keras.utils.get_file(
    'auto-mpg.data',
    'http://archive.isc.uci.edu/ml/machine-learning-datasets/auto-mpg/auto-mpg/data'
)

column_names = [
                'MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin'
                ]

raw_dataset = pd.read_csv[
    dataset_path,
    names=column_names,
    na_values='?',
    comment='\t',
    sep=' ',
    skipinitialspace=True
]

df = raw_dataset.copy()

# Checking for na values in dataframe
df.isna().sum()
# Dropping na
df1 = df.dropna()

# OR can impute na with a value
mean_hp = df['Horsepower'].mean()
df2 = df.fillna(mean_hp)

df = df2.copy()

# Map integer values to strings where they're categorical
df['Origin'] = df['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))

df = pd.get_dummies(df, prefix='', prefix_sep='')

# Randomly collects a number of items to use for training
train_dataset = df.sample(frac=0.8, rnadom_state=0)
# removes all the indices from the training dataset, leaving just the test dataset
test_dataset = df.drop(train_dataset.index)

# Removes labels from train dataset
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# USING ZIP AND MAP

dataset1 = tf.data.Dataset.from_tensor_slice(tf.random.uniform([4.10]))

dataset2 = tf.data.Dataset.from_tensor_slices(
    (
        tf.random.uniform([4]),
        tf.random.uniform([4,100], maxval=100, dtype=tf.int32)
    )
)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

# Padding these batches to make them the same size
dataset4 = tf.data.Dataset.range(100)
dataset4 = dataset4.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
