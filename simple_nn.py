import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only show error messages (suppress warnings)

import tensorflow as tf
import numpy as np

# Check if TensorFlow is installed properly
try:
    tf_version = tf.__version__
    print("TensorFlow version:", tf_version)
    print("TensorFlow is installed properly.")
except ImportError:
    print("TensorFlow is not installed properly. Please install TensorFlow before running this script.")
    exit()

# Define a small neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=5, activation='relu'),
    tf.keras.layers.Dense(units=3)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits='true'),
              metrics=['accuracy'])

# Dummy data for training
X_train = np.random.rand(100, 4)
y_train = np.random.randint(0, 3, size=(100,))

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16)
print("\nTraining complete. Running an inference.")

# Test the model with dummy data
X_test = np.random.rand(10, 4)
y_pred = model.predict(X_test)
print("Predictions:\n", y_pred)