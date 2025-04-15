import os
import argparse
import numpy as np
import flwr as fl
import tensorflow as tf
from logging import INFO, basicConfig
from flwr.common.logger import log

from flwr_implementation import model
from flwr_implementation import dataset
from flwr_implementation.kafka_client import start_kafka_client

# Configure logging
basicConfig(level=INFO)

class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Return properties for this client."""
        return {}

    def get_parameters(self):
        """Return current model parameters."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train model on local data."""
        # Set model parameters
        self.model.set_weights(parameters)

        # Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,  # Local training for just 1 epoch
            batch_size=32,
            verbose=0  # Don't print training progress
        )

        # Return updated model parameters and training stats
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        # Set model parameters
        self.model.set_weights(parameters)

        # Evaluate the model
        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0
        )

        # Return evaluation metrics
        return loss, len(self.x_test), {"accuracy": accuracy}


def main():
    """Parse arguments and start client."""
    parser = argparse.ArgumentParser(description="Flower client using Kafka")
    parser.add_argument(
        "--broker",
        type=str,
        default="localhost:9094",
        help="Kafka broker address in the format host:port"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Client ID (will be randomly generated if not provided)"
    )
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Use gRPC instead of Kafka"
    )

    args = parser.parse_args()

    # Generate client ID if not provided
    client_id = args.client_id if args.client_id else str(np.random.randint(0, 10000))

    # Create and compile Keras model
    keras_model = model.create_keras_model()
    keras_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Load local data partition
    partition_id = int(client_id) % 10  # Use client ID to determine partition
    (x_train, y_train), (x_test, y_test) = dataset.load_partition(partition_id)

    # Create Flower client
    client = CifarClient(keras_model, x_train, y_train, x_test, y_test)

    # Start Flower client
    log(INFO, f"Starting client {client_id} with broker {args.broker}")
    log(INFO, f"Using {'gRPC' if args.grpc else 'Kafka'} for communication")

    if args.grpc:
        # Start client with gRPC
        fl.client.start_numpy_client(
            server_address=args.broker,
            client=client
        )
    else:
        # Start client with our custom Kafka implementation
        start_kafka_client(
            broker=args.broker,
            numpy_client=client,
            clientid=client_id
        )


if __name__ == "__main__":
    main()
