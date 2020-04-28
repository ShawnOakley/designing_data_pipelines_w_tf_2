# Migratinng from tf 1.0
# Loading data
# Prepping data
# Optimizing data pipeline performance
# Generally understanding tf.data module

# Improvements from TF 1.0 to 2.0
# 1) Easier model building
#     Combination into Keras API
#     Eager execution
# 2) Robust deployment into production with tfx framework
#       https://www.tensorflow.org/tfx/tutorials
# 3) Improved utility libraries
# 4) Improved Data Pipelines

# tf.data features
# Handles large amounts of data
# Easier deployments
# Simplified API
# Performance Optimizations
# - Read from different file formats
# - Complex transformations
# - Easy reading and parallelizations

# Migrating from TF 1.0
# tensorflow.org/guide/migrate

# Example:
import tensorflow as tf
import numpy as np
# Can import keras directly or just do tf.keras
# from tensorflow import keras

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images.shape

train_images = train_images / 255
test_images = test_images / 255

# Creating keras model of Sequential Layers
# Flatten is entrypoint
# Then Dense, with 128 shape and activation function of relu
# Final step is exit point of Dense, with 10 categories and softmax(?)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metric=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])