"""
Federated Learning Client

This module implements a client for federated learning using Kafka for communication.
The client receives a global model from the server, trains it on local data,
and sends the updated model back to the server.
"""

import os
import time
import signal
import sys
import numpy as np
import gc

# Import common modules
from common.logger import get_client_logger
from common.model import create_cifar10_model, are_weights_compatible, adapt_weights
from common.data import load_cifar10_data
from common.serialization import serialize_weights, deserialize_weights
from common.kafka_utils import create_producer, create_consumer, send_message, receive_messages, close_kafka_resources

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class FederatedClient:
    """Client for federated learning"""

    def __init__(self, bootstrap_servers, update_topic, model_topic, client_id=None):
        """
        Initialize the federated learning client.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            update_topic: Topic for sending model updates to the server
            model_topic: Topic for receiving global model from the server
            client_id: Client ID (optional, will be generated if not provided)
        """
        # Set client ID
        self.client_id = int(client_id) if client_id is not None else self._generate_client_id()

        # Initialize logger
        self.logger = get_client_logger(self.client_id)
        self.logger.info(f"Initializing client {self.client_id}")

        # Set Kafka configuration
        self.bootstrap_servers = bootstrap_servers
        self.update_topic = update_topic
        self.model_topic = model_topic
        self.producer = None
        self.consumer = None

        # Initialize model and data
        self.model = None
        self.X_train, self.y_train, self.X_test, self.y_test = load_cifar10_data(
            subset_size=5000, test_size=1000, logger=self.logger
        )

        # Connect to Kafka
        self._connect_to_kafka()

    def _generate_client_id(self):
        """Generate a unique client ID based on timestamp"""
        return int(time.time() * 1000) % 10000

    def _connect_to_kafka(self):
        """Connect to Kafka broker"""
        self.logger.info(f"Connecting to Kafka at {self.bootstrap_servers}")

        # Create producer
        self.producer = create_producer(
            bootstrap_servers=self.bootstrap_servers,
            logger=self.logger
        )

        # Create consumer
        self.consumer = create_consumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id=f"federated_client_{self.client_id}",
            topics=[self.model_topic],
            logger=self.logger
        )

        if self.producer is not None and self.consumer is not None:
            self.logger.info("Successfully connected to Kafka")
            return True
        else:
            self.logger.error("Failed to connect to Kafka")
            return False

    def receive_global_model(self):
        """
        Receive the global model from the server.

        Returns:
            List of weight arrays, or None if reception fails
        """
        self.logger.info(f"Waiting for global model from server on topic {self.model_topic}")

        # Receive messages from the model topic
        messages = receive_messages(
            consumer=self.consumer,
            timeout_ms=60000,
            max_messages=1,
            logger=self.logger
        )

        if not messages:
            self.logger.error("No global model received from server")
            return None

        # Deserialize the global model
        global_model = deserialize_weights(messages[0], logger=self.logger)

        if global_model is not None:
            self.logger.info(f"Received global model with {len(global_model)} layers")

        return global_model

    def train_local_model(self, global_weights):
        """
        Train the local model using the global weights.

        Args:
            global_weights: Global model weights

        Returns:
            Updated model weights, or None if training fails
        """
        self.logger.info("Training local model")

        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Simulating training.")
            # Simulate training by adding noise to the global weights
            local_weights = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in global_weights]
            self.model = local_weights
            return local_weights

        try:
            # Run garbage collection before training
            gc.collect()

            # Create model
            model = create_cifar10_model()

            if model is None:
                self.logger.error("Failed to create model")
                return None

            # Check if weights are compatible
            if not are_weights_compatible(model, global_weights):
                self.logger.warning("Global weights are not compatible with the model. Attempting to adapt.")
                adapted_weights = adapt_weights(model, global_weights)

                if adapted_weights is None:
                    self.logger.error("Could not adapt weights. Training with default initialization.")
                else:
                    self.logger.info("Successfully adapted weights")
                    model.set_weights(adapted_weights)
            else:
                model.set_weights(global_weights)

            # Train the model
            self.logger.info("Training model on local data")

            # Use a subset of data for training
            train_size = min(5000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), train_size, replace=False)
            X_subset = self.X_train[indices]
            y_subset = self.y_train[indices]

            # Define callbacks
            class LoggingCallback(tf.keras.callbacks.Callback):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger

                def on_epoch_end(self, epoch, logs=None):
                    self.logger.info(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")

        
            # Train for multiple epochs with smaller batch size
            model.fit(
                X_subset, y_subset,
                epochs=5,
                batch_size=32,  # Reduced batch size to save memory
                verbose=0,
                validation_split=0.2,
                callbacks=[LoggingCallback(self.logger)]
            )

            # Evaluate the model
            test_size = min(1000, len(self.X_test))
            test_indices = np.random.choice(len(self.X_test), test_size, replace=False)
            X_test_subset = self.X_test[test_indices]
            y_test_subset = self.y_test[test_indices]

            loss, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)
            self.logger.info(f"Model evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")

            # Get the updated weights
            local_weights = model.get_weights()
            self.model = local_weights

            # Run garbage collection after training
            gc.collect()

            return local_weights
        except Exception as e:
            self.logger.error(f"Error training local model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def send_model_update(self, weights):
        """
        Send the updated model to the server.

        Args:
            weights: Updated model weights

        Returns:
            True if the update was sent successfully, False otherwise
        """
        self.logger.info(f"Sending model update to server on topic {self.update_topic}")

        # Serialize the weights
        serialized_weights = serialize_weights(weights, logger=self.logger)

        if serialized_weights is None:
            self.logger.error("Failed to serialize model weights")
            return False

        # Send the update
        success = send_message(
            producer=self.producer,
            topic=self.update_topic,
            message=serialized_weights,
            logger=self.logger
        )

        if success:
            self.logger.info("Model update sent successfully")
        else:
            self.logger.error("Failed to send model update")

        return success

    def start(self, num_rounds=10):
        """
        Start the federated learning process.

        Args:
            num_rounds: Number of federated learning rounds (default: 10)
        """
        self.logger.info(f"Starting federated learning with {num_rounds} rounds")
        
        start_time = time.time()
        final_loss = None
        final_accuracy = None

        for round_idx in range(num_rounds):
            self.logger.info(f"=== Round {round_idx + 1}/{num_rounds} ===")

            # Receive global model
            global_weights = self.receive_global_model()

            if global_weights is None:
                self.logger.error(f"Failed to receive global model in round {round_idx + 1}")
                continue

            # Train local model
            local_weights = self.train_local_model(global_weights)

            if local_weights is None:
                self.logger.error(f"Failed to train local model in round {round_idx + 1}")
                continue

            # Store the final metrics from the last successful training
            if TENSORFLOW_AVAILABLE:
                model = create_cifar10_model()
                model.set_weights(local_weights)
                test_size = min(1000, len(self.X_test))
                test_indices = np.random.choice(len(self.X_test), test_size, replace=False)
                X_test_subset = self.X_test[test_indices]
                y_test_subset = self.y_test[test_indices]
                final_loss, final_accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)

            # Send model update
            success = self.send_model_update(local_weights)

            if not success:
                self.logger.error(f"Failed to send model update in round {round_idx + 1}")
                continue

            self.logger.info(f"Completed round {round_idx + 1}/{num_rounds}")

        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info("Federated learning completed")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        if final_loss is not None and final_accuracy is not None:
            self.logger.info(f"Final metrics - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

    def close(self):
        """Close resources"""
        
        self.logger.info("Closing client resources")

        # Close Kafka resources
        close_kafka_resources(
            producer=self.producer,
            consumer=self.consumer,
            logger=self.logger
        )

        # Clear data
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.logger.info("Client resources closed")

# Global variable for signal handling
client_instance = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("Received termination signal. Closing client gracefully...")
    if client_instance is not None:
        client_instance.close()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration from environment variables
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    client_id = os.environ.get("CLIENT_ID")

    # Create and start client
    client = FederatedClient(
        bootstrap_servers=bootstrap_servers,
        update_topic="model_updates",
        model_topic="global_model",
        client_id=client_id
    )

    # Store client instance for signal handling
    client_instance = client

    # Start federated learning
    client.start(num_rounds=10)

    # Close resources
    client.close()
