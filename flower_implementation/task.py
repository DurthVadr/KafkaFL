"""flower-benchmark: A Flower / TensorFlow app."""

import os

import keras
from keras import layers
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras.models import Sequential
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU-only operation
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available in model.py")
    TENSORFLOW_AVAILABLE = False



# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # define lenet model for CIFAR-10
    model = Sequential([
        # First convolutional layer
        Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
        AveragePooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        Conv2D(16, kernel_size=(5, 5), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),

            # Fully connected layers
        Flatten(),
        Dense(120, activation='relu'),
        Dropout(0.2),
        Dense(84, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

     
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test
