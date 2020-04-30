# Two ways to create a tf.data.Dataset
# 1 --  Data Source, Construct a Dataset from data stored in memory or in file
# 2 -- Data Transformation, Construct Dataset from one or more data objects

import numpy as np
import pandas as pd
import tensorflow as tf

# CSV

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.cc"
TEST_DATA_URL = "https://storage.googleapis.com/tf=datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

LABEL_COLUMN = "survived"

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value='?',
        num_epoch=1,
        ignore_errors=True,
        **kwargs)

    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, valyue in batch.items():
            print(f'{key}, {value.numpy()}')

show_batch(raw_test_data)

CSV_COLUMNS = [
    'survived',
    'sex',
    'age',
    'n_siblings_spouses',
    'parch',
    'fare',
    'class',
    'deck',
    'embark_town',
    'town'
]

temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)

show_batch(temp_dataset)
CSV_COLUMNS = [
    'survived',
    'class',
    'deck',
    'embark_town',
    'alone'
]

temp_dataset  = get_dataset(train_file_path, select_columns=CSV_COLUMNS)
show_batch(temp_dataset)

# PANDAS - useful for tabular data, not for image or text data

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
with np.load(path) as data:
    train_examples = data['x_train']
    train_labels = data['y_train']
    test_examples = data['x_test']
    test_labels = data['y_test']

print(train_examples)
type(train_examples)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

type(train_dataset)

for values in train_dataset.take(1):
    print(values)

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')

