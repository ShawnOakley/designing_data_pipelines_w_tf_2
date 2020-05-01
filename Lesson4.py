# Performance Optimizations
# GPUs and TPUs can drastically reduce the time
# required to execute a single training step.
# Achieving peak performance requires an efficient pipeline
# that delivers daya for the next step before the current
# step has finished.

import pandas as pd
import tensorflow as tf
from tensorflow import keras
import time

csv_file = tf.keras.utils.get_file('heart.csv',
                                    'https://storage.googleapis.com/applied-dl/heart.csv')
df = pd.read_csv(csv_file)

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = pd.thal.cat.codes


train_dataset = df.sample(frac=0.8,
                          random_state=0)
test_dataset = df.drop(train_dataset.index)
train_labels = train_dataset.pop('target')
test_labels = test.dataset.pop('target')

# normalize values
def norm(x, train_stats):
    return (x-train_stats['mean'])/train_stats['std']

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

# Normalize train data
normed_train_data = norm(train_dataset, train_stats)
# Normalize test data
normed_test_data  = norm(test_dataset, train_stats)

# Build out Keras Sequential API
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(normed_train_data.keys())]),
    keras.layers.Dense(64, activation='relu')
    keras.layers.Dense(1)
])

optimizer = model.optimizers.RMSprop(0.001)
model.compile(loss='mse',
              optimize=optimizer,
              metrics=['mse', 'mae'])

model.fit(normed_train_data, train_labels, epochs=10)

y_hats = model.predict(normed_test_data)
y_hats = [x[0] for x in y_hats]
keras.losses.MAE(
    test_labels, y_hats
)

model.evaluate(normed_test_data, test_labels)

# BATCHING AND PREFETCHING
# Batching: split dataset up into multiple pieces
# Prefetching: reads data off of disk before needed by model.
# Alleviates bottleneck

# batch by adding the batch method
# prefetch 2 gives you 2 batches at a time, so next batch
# is already loaded
dataset = tf.data.Dataset.from_tensor_slices(
    (normed_train_data.values, train_labels.values)
).batch(100).prefetch(2)
# Each epoch is a full evaluation of all the batches
model.fit(dataset, epoch=100)

# PARALLELIZING DATA EXTRACTION
# Data stored remotely requires different processing
# than data stored locally

# Time-to-first-byte: Reading the first byte of data
# from a remote storage location can take orders of
# magnitude longer than from local storage
# Read throughput: Remote storage can offer large bandwidth
# but reading a single file might only be able
# to use a fraction of this bandwidth

# Interleave: Tensorflow transformation used to parallelize
# the data loading step
# Can specify the number of datasets to overlap (the interleaving), and the level of parallelism

# EXAMPLE:
tf.data.Dataset.interleave(
    dataset,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)

# Parallelizing interleave
# Uses 'num_parallel_calls' argument to specify the level of parallelism
# and load multiple datasets in parallel
# 'tf.data.experimental.AUTOTUNE' delegates the decision
# about what level of parallelism to use at runtime

# BEST PRACTICES FOR PIPELINE PERFORMANCE
# .batch -- Separate data into smaller btches rather than a
# single dataset
# .prefetch -- Overlap the work of loading data and model training
# .interleave -- Make data reads and transformations run in parallel