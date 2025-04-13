import tensorflow as tf
import numpy as np

def load_partition(idx):
    """Load 1/10 of the CIFAR-10 training and test data to simulate a partition."""
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Flatten labels
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    # Create partitions (10 partitions)
    n_partitions = 10
    # Use modulo to determine which examples go to which partition
    train_partition_idx = np.arange(len(x_train)) % n_partitions
    test_partition_idx = np.arange(len(x_test)) % n_partitions
    
    # Get partition for this client
    x_train_partition = x_train[train_partition_idx == idx]
    y_train_partition = y_train[train_partition_idx == idx]
    x_test_partition = x_test[test_partition_idx == idx]
    y_test_partition = y_test[test_partition_idx == idx]
    
    return (x_train_partition, y_train_partition), (x_test_partition, y_test_partition)
