"""
Lightweight model definition for federated learning.
This model uses fewer parameters to reduce memory usage.
"""

import logging
import numpy as np

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU-only operation
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available in lightweight_model.py")
    TENSORFLOW_AVAILABLE = False

def create_lightweight_model():
    """
    Create a lightweight model for CIFAR-10 classification.
    Uses fewer parameters to reduce memory usage.

    Returns:
        A compiled Keras model if TensorFlow is available, None otherwise
    """
    if not TENSORFLOW_AVAILABLE:
        logging.warning("TensorFlow not available. Cannot create model.")
        return None

    try:
        # Very simple CNN for CIFAR-10
        model = Sequential([
            # First convolutional layer
            Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(pool_size=(2, 2)),

            # Second convolutional layer
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            # Fully connected layers
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        # Compile model with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        logging.error(f"Error creating lightweight model: {e}")
        return None

def get_lightweight_random_weights():
    """
    Generate random weights for the lightweight model.

    Returns:
        List of numpy arrays with random weights
    """
    if TENSORFLOW_AVAILABLE:
        # Create model and get its weight shapes
        model = create_lightweight_model()
        if model is None:
            return [np.random.rand(10).astype(np.float32)]

        # Initialize with random weights
        weights = [np.random.normal(0, 0.05, w.shape).astype(np.float32) for w in model.get_weights()]
        return weights
    else:
        # Approximate shapes for the lightweight model
        shapes = [
            # First conv block
            (3, 3, 3, 16), (16,),  # Conv2D

            # Second conv block
            (3, 3, 16, 32), (32,),  # Conv2D

            # Fully connected layers
            (8 * 8 * 32, 64), (64,),  # Dense
            (64, 10), (10,)  # Output layer
        ]

        # Create random weights with the appropriate shapes
        weights = [np.random.normal(0, 0.05, shape).astype(np.float32) for shape in shapes]
        return weights
