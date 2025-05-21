"""
Federated Learning Server

This module implements a server for federated learning using Kafka for communication.
The server aggregates model updates from clients and distributes the global model.
"""

import os
import time
import signal
import sys
import numpy as np
import gc

# Import common modules
from common.logger import get_server_logger
from common.model import create_lenet_model, get_random_weights
from common.serialization import serialize_weights, deserialize_weights
from common.kafka_utils import create_producer, create_consumer, send_message, receive_messages, close_kafka_resources

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU-only operation
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

    # Limit TensorFlow to use only necessary memory
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class FederatedServer:
    """Server for federated learning"""

    def __init__(self, bootstrap_servers, model_topic, update_topic):
        """
        Initialize the federated learning server.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            model_topic: Topic for sending global model to clients
            update_topic: Topic for receiving model updates from clients
        """
        # Initialize logger
        self.logger = get_server_logger()
        self.logger.info("Initializing federated learning server")

        # Set Kafka configuration
        self.bootstrap_servers = bootstrap_servers
        self.model_topic = model_topic
        self.update_topic = update_topic
        self.producer = None
        self.consumer = None

        # Initialize global model
        self.global_model = self._initialize_global_model()

        # Connect to Kafka
        self._connect_to_kafka()

    def _initialize_global_model(self):
        """
        Initialize the global model.

        Returns:
            List of weight arrays representing the global model
        """
        self.logger.info("Initializing global model (LeNet)")

        if TENSORFLOW_AVAILABLE:
            # Create LeNet model
            model = create_lenet_model()

            if model is not None:
                # Get initial weights
                weights = model.get_weights()
                self.logger.info(f"Initialized LeNet model with {len(weights)} layers")

                # Log the shapes of the weights for debugging
                for i, w in enumerate(weights):
                    self.logger.info(f"Layer {i} shape: {w.shape}")

                return weights

        # Fallback to random weights
        self.logger.warning("Using random weights for LeNet model")
        weights = get_random_weights(model_type="lenet")
        self.logger.info(f"Initialized random LeNet model with {len(weights)} layers")

        # Log the shapes of the weights for debugging
        for i, w in enumerate(weights):
            self.logger.info(f"Layer {i} shape: {w.shape}")

        return weights

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
            group_id="federated_server",
            topics=[self.update_topic],
            logger=self.logger
        )

        if self.producer is not None and self.consumer is not None:
            self.logger.info("Successfully connected to Kafka")
            return True
        else:
            self.logger.error("Failed to connect to Kafka")
            return False

    def send_global_model(self):
        """
        Send the global model to clients.

        Returns:
            True if the model was sent successfully, False otherwise
        """
        self.logger.info(f"Sending global model to clients on topic {self.model_topic}")

        # Serialize the global model
        serialized_model = serialize_weights(self.global_model, logger=self.logger)

        if serialized_model is None:
            self.logger.error("Failed to serialize global model")
            return False

        # Send the global model
        success = send_message(
            producer=self.producer,
            topic=self.model_topic,
            message=serialized_model,
            logger=self.logger
        )

        if success:
            self.logger.info("Global model sent successfully")
        else:
            self.logger.error("Failed to send global model")

        return success

    def receive_model_updates(self, max_updates=3, timeout_ms=60000):
        """
        Receive model updates from clients.

        For asynchronous operation, use a short timeout (e.g., 1000ms) and small max_updates
        to poll for updates without blocking for too long.

        Args:
            max_updates: Maximum number of updates to receive (default: 3)
            timeout_ms: Timeout in milliseconds (default: 60000)

        Returns:
            List of model updates, or empty list if no updates were received
        """
        # Log based on timeout duration
        if timeout_ms < 5000:
            self.logger.debug(f"Polling for model updates (max: {max_updates})")
        else:
            self.logger.info(f"Waiting for model updates from clients on topic {self.update_topic}")

        # Receive messages from the update topic
        messages = receive_messages(
            consumer=self.consumer,
            timeout_ms=timeout_ms,
            max_messages=max_updates,
            logger=self.logger
        )

        if not messages:
            if timeout_ms >= 5000:
                self.logger.warning("No model updates received from clients")
            return []

        # Deserialize model updates
        updates = []
        for i, message in enumerate(messages):
            update = deserialize_weights(message, logger=self.logger)
            if update is not None:
                updates.append(update)

        if updates:
            self.logger.info(f"Received {len(updates)} valid model updates from clients")

        return updates

    def aggregate_model_updates(self, updates):
        """
        Aggregate model updates using federated averaging.

        Args:
            updates: List of model updates

        Returns:
            Aggregated model weights
        """
        if not updates:
            self.logger.warning("No updates to aggregate. Keeping current global model.")
            return self.global_model

        self.logger.info(f"Aggregating {len(updates)} model updates")

        # Run garbage collection before aggregation
        gc.collect()

        # Get the number of layers in the model
        num_layers = len(self.global_model)

        # Check if all updates have the same number of layers
        if not all(len(update) == num_layers for update in updates):
            self.logger.error("Updates have different numbers of layers. Cannot aggregate.")
            return self.global_model

        # Initialize aggregated weights
        aggregated_weights = []

        try:
            # Aggregate each layer
            for layer_idx in range(num_layers):
                # Get weights for this layer from all updates
                layer_weights = [update[layer_idx] for update in updates]

                # Check if all weights have the same shape
                if not all(w.shape == layer_weights[0].shape for w in layer_weights):
                    self.logger.error(f"Layer {layer_idx} has inconsistent shapes across updates.")
                    return self.global_model

                # Average the weights
                layer_avg = np.mean(layer_weights, axis=0)
                aggregated_weights.append(layer_avg)

                # Free memory
                del layer_weights

            self.logger.info(f"Successfully aggregated model updates")

            # Run garbage collection after aggregation
            gc.collect()

            return aggregated_weights
        except Exception as e:
            self.logger.error(f"Error aggregating model updates: {e}")
            return self.global_model

    def start(self, duration_minutes=60, aggregation_interval_seconds=60, min_updates_per_aggregation=1):
        """
        Start the federated learning process with time-based asynchronous aggregation.

        Args:
            duration_minutes: Total duration to run the server in minutes (default: 60)
            aggregation_interval_seconds: Interval between model aggregations in seconds (default: 60)
            min_updates_per_aggregation: Minimum number of updates required for aggregation (default: 1)
        """
        self.logger.info(f"Starting federated learning server (duration: {duration_minutes}m, interval: {aggregation_interval_seconds}s)")

        # Send initial global model
        if not self.send_global_model():
            self.logger.error("Failed to send initial global model. Exiting.")
            return

        # Initialize tracking variables
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_aggregation_time = start_time
        aggregation_count = 0
        total_updates_received = 0
        pending_updates = []

        try:
            while time.time() < end_time:
                current_time = time.time()

                # Log progress every minute
                if int(current_time) % 60 == 0:
                    time_elapsed_minutes = (current_time - start_time) / 60
                    time_remaining_minutes = duration_minutes - time_elapsed_minutes
                    self.logger.info(f"Server running: {time_elapsed_minutes:.1f}m elapsed, {time_remaining_minutes:.1f}m remaining")

                # Check if it's time to aggregate
                time_since_last_aggregation = current_time - last_aggregation_time
                should_aggregate = (time_since_last_aggregation >= aggregation_interval_seconds and
                                   len(pending_updates) >= min_updates_per_aggregation)

                if should_aggregate:
                    self.logger.info(f"=== Aggregation {aggregation_count + 1} ===")

                    # Aggregate pending updates
                    self.global_model = self.aggregate_model_updates(pending_updates)

                    # Send updated global model to clients
                    self.send_global_model()

                    # Update tracking variables
                    last_aggregation_time = current_time
                    aggregation_count += 1
                    total_updates_received += len(pending_updates)
                    pending_updates = []

                    # Run garbage collection after aggregation
                    gc.collect()

                # Receive any available updates (non-blocking)
                new_updates = self.receive_model_updates(max_updates=10, timeout_ms=1000)
                if new_updates:
                    pending_updates.extend(new_updates)

                # Sleep briefly to avoid tight polling
                time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Server interrupted by user")

        # Final statistics
        self.logger.info(f"Federated learning completed: {aggregation_count} aggregations, {total_updates_received} updates")

        # Perform one final aggregation if there are pending updates
        if pending_updates:
            self.logger.info(f"Performing final aggregation with {len(pending_updates)} updates")
            self.global_model = self.aggregate_model_updates(pending_updates)
            self.send_global_model()

    def close(self):
        """Close resources"""
        self.logger.info("Closing server resources")

        # Close Kafka resources
        close_kafka_resources(
            producer=self.producer,
            consumer=self.consumer,
            logger=self.logger
        )

        # Clear model
        self.global_model = None

        self.logger.info("Server resources closed")

# Global variable for signal handling
server_instance = None

def signal_handler(*_):
    """Handle termination signals"""
    print("Received termination signal. Closing server gracefully...")
    if server_instance is not None:
        server_instance.close()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration from environment variables
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    duration_minutes = int(os.environ.get("DURATION_MINUTES", "60"))
    aggregation_interval = int(os.environ.get("AGGREGATION_INTERVAL_SECONDS", "60"))
    min_updates = int(os.environ.get("MIN_UPDATES_PER_AGGREGATION", "1"))

    # Create and start server
    server = FederatedServer(
        bootstrap_servers=bootstrap_servers,
        model_topic="global_model",
        update_topic="model_updates"
    )

    # Store server instance for signal handling
    server_instance = server

    try:
        # Start federated learning
        server.start(
            duration_minutes=duration_minutes,
            aggregation_interval_seconds=aggregation_interval,
            min_updates_per_aggregation=min_updates
        )
    finally:
        # Close resources
        server.close()
