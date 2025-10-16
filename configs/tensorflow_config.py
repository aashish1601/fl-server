"""
Configuration for TensorFlow/Keras Models
Example: MNIST with Keras Sequential model
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ============================================
# MODEL DEFINITION (TensorFlow/Keras)
# ============================================
def create_keras_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN in Keras"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_CLASS = create_keras_model  # Function that creates model
MODEL_KWARGS = {
    'input_shape': (28, 28, 1),
    'num_classes': 10
}

# Specify framework
FRAMEWORK = 'tensorflow'

# ============================================
# TRAINING CONFIGURATION
# ============================================
OPTIMIZER = keras.optimizers.Adam(learning_rate=0.001)
LOSS = keras.losses.SparseCategoricalCrossentropy()
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 128

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """
    Load MNIST data for TensorFlow
    Returns: (train_data, test_data) as tuples (x, y)
    """
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Reshape for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Split data between clients
    if client_id == 0:
        train_indices = slice(0, len(x_train) // 2)
        test_indices = slice(0, len(x_test) // 2)
    else:
        train_indices = slice(len(x_train) // 2, len(x_train))
        test_indices = slice(len(x_test) // 2, len(x_test))
    
    train_data = (x_train[train_indices], y_train[train_indices])
    test_data = (x_test[test_indices], y_test[test_indices])
    
    return train_data, test_data


# ============================================
# NOTE: To use with generic system
# ============================================
"""
This config works with TensorFlow models!

Usage:
    # Server
    python server_generic.py --config configs/tensorflow_config.py
    
    # Client  
    python client_generic.py --config configs/tensorflow_config.py

The framework adapter will automatically detect TensorFlow and use
appropriate methods for parameter extraction and training.
"""


