"""
Data handling utilities for federated learning system.
Provides functions for loading and preprocessing data.
"""

import logging
import numpy as np
import os

# Check if reduced data size should be used
REDUCED_DATA_SIZE = os.environ.get("REDUCED_DATA_SIZE", "0") == "1"

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU-only operation
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

    from tensorflow.keras.datasets import cifar10
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available in data.py")
    TENSORFLOW_AVAILABLE = False

def load_cifar10_data(subset_size=50000, test_size=10000, logger=None):
    """
    Load and preprocess CIFAR-10 data.
    Uses reduced data size if specified in environment variable.

    Args:
        subset_size: Number of training samples to use (default: 5000)
        test_size: Number of test samples to use (default: 1000)
        logger: Logger instance for logging (optional)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    # Load CIFAR-10 data first to get actual sizes
    if TENSORFLOW_AVAILABLE:
        (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
        max_train_size = len(X_train_full)  # 50,000
        max_test_size = len(X_test_full)    # 10,000
    else:
        max_train_size = 5000
        max_test_size = 1000

    # Set default sizes based on reduced data flag
    if REDUCED_DATA_SIZE:
        subset_size = subset_size or 1000    # Reduced: 1000 samples
        test_size = test_size or 200         # Reduced: 200 samples
        if logger:
            logger.info(f"Using reduced data size: {subset_size} training samples, {test_size} test samples")
    else:
        subset_size = subset_size or max_train_size  # Full: 50,000 samples
        test_size = test_size or max_test_size       # Full: 10,000 samples
        if logger:
            logger.info(f"Using FULL dataset: {subset_size} training samples, {test_size} test samples")

    # Use random data if TensorFlow is not available
    if not TENSORFLOW_AVAILABLE:
        if logger:
            logger.warning("TensorFlow not available. Using random data instead of CIFAR-10.")
        return _create_random_data(subset_size, test_size)

    try:
        # Load CIFAR-10 data
        (X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

        # Preprocess data
        X_train_full = X_train_full.astype('float32') / 255.0
        y_train_full = np.squeeze(y_train_full)
        X_test_full = X_test_full.astype('float32') / 255.0
        y_test_full = np.squeeze(y_test_full)

        # Use subsets for faster training and evaluation
        X_train = X_train_full[:subset_size]
        y_train = y_train_full[:subset_size]
        X_test = X_test_full[:test_size]
        y_test = y_test_full[:test_size]

        if logger:
            logger.info(f"Loaded CIFAR-10 data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")

        return X_train, y_train, X_test, y_test
    except Exception as e:
        if logger:
            logger.error(f"Error loading CIFAR-10 data: {e}")

        # Fallback to random data
        return _create_random_data(subset_size, test_size)

def _create_random_data(train_size, test_size):
    """Create random data with the same shape as CIFAR-10"""
    X_train = np.random.rand(train_size, 32, 32, 3).astype('float32')
    y_train = np.random.randint(0, 10, size=train_size)
    X_test = np.random.rand(test_size, 32, 32, 3).astype('float32')
    y_test = np.random.randint(0, 10, size=test_size)
    return X_train, y_train, X_test, y_test

def get_data_batch(X, y, batch_size=32, random=True):
    """
    Get a batch of data for training.

    Args:
        X: Input data
        y: Target data
        batch_size: Size of the batch (default: 32)
        random: Whether to select random samples (default: True)

    Returns:
        Tuple of (X_batch, y_batch)
    """
    # Ensure batch size doesn't exceed data size
    batch_size = min(batch_size, len(X))

    if random:
        # Select random indices without replacement
        indices = np.random.choice(len(X), batch_size, replace=False)
    else:
        # Select first batch_size samples
        indices = np.arange(batch_size)

    return X[indices], y[indices]
