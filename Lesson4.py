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