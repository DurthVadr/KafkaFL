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
from common.model import create_lenet_model, are_weights_compatible, adapt_weights
from common.data import load_cifar10_data
from common.serialization import serialize_weights, deserialize_weights
from common.kafka_utils import create_producer, create_consumer, send_message, receive_messages, close_kafka_resources
from common.visualization import plot_client_accuracy, plot_client_loss

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

        # Initialize metrics tracking
        self.metrics = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_loss': [],
            'training_times': []
        }
        self.best_accuracy = 0.0

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
        training_start_time = time.time()

        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow not available. Simulating training.")
            # Simulate training by adding noise to the global weights
            local_weights = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in global_weights]
            self.model = local_weights

            # Add simulated metrics
            self.metrics['train_accuracy'].append(0.5)  # Simulated accuracy
            self.metrics['test_accuracy'].append(0.5)   # Simulated accuracy
            self.metrics['train_loss'].append(1.0)      # Simulated loss
            self.metrics['training_times'].append(time.time() - training_start_time)

            return local_weights

        # Run garbage collection before training
        gc.collect()

        # Create LeNet model
        model = create_lenet_model()
        if model is None:
            self.logger.error("Failed to create LeNet model")
            return None

        try:
            # Check if weights are compatible
            if not are_weights_compatible(model, global_weights):
                self.logger.warning("Global weights are not compatible. Attempting to adapt.")
                adapted_weights = adapt_weights(model, global_weights)
                if adapted_weights is None:
                    self.logger.error("Could not adapt weights. Using default initialization.")
                else:
                    self.logger.info("Successfully adapted weights")
                    model.set_weights(adapted_weights)
            else:
                model.set_weights(global_weights)

            # Use a subset of data for training
            train_size = min(5000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), train_size, replace=False)
            X_subset = self.X_train[indices]
            y_subset = self.y_train[indices]

            # Define metrics tracking callback
            class MetricsCallback(tf.keras.callbacks.Callback):
                def __init__(self, client):
                    super().__init__()
                    self.client = client
                    self.logger = client.logger
                    self.epoch_metrics = {'loss': [], 'accuracy': []}

                def on_epoch_end(self, epoch, logs=None):
                    self.logger.info(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")
                    self.epoch_metrics['loss'].append(logs['loss'])
                    self.epoch_metrics['accuracy'].append(logs['accuracy'])

            metrics_callback = MetricsCallback(self)

            # Train the model
            history = model.fit(
                X_subset, y_subset,
                epochs=5,
                batch_size=32,
                verbose=0,
                validation_split=0.2,
                callbacks=[metrics_callback]
            )

            # Store training metrics
            final_train_loss = metrics_callback.epoch_metrics['loss'][-1]
            final_train_accuracy = metrics_callback.epoch_metrics['accuracy'][-1]
            self.metrics['train_loss'].append(final_train_loss)
            self.metrics['train_accuracy'].append(final_train_accuracy)

            # Evaluate the model
            test_size = min(1000, len(self.X_test))
            test_indices = np.random.choice(len(self.X_test), test_size, replace=False)
            X_test_subset = self.X_test[test_indices]
            y_test_subset = self.y_test[test_indices]

            loss, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)
            self.logger.info(f"Model evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")

            # Store test metrics
            self.metrics['test_accuracy'].append(accuracy)
            self.metrics['training_times'].append(time.time() - training_start_time)

            # Update best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.logger.info(f"New best accuracy: {self.best_accuracy:.4f}")

            # Get the updated weights
            local_weights = model.get_weights()
            self.model = local_weights

            # Run garbage collection after training
            gc.collect()

            return local_weights
        except Exception as e:
            self.logger.error(f"Error training local model: {e}")
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

    def start(self, duration_minutes=60, training_interval_seconds=120):
        """
        Start the asynchronous federated learning process.

        The client will continuously check for updated global models, train on them,
        and send updates back to the server for the specified duration.

        Args:
            duration_minutes: Total duration to run the client in minutes (default: 60)
            training_interval_seconds: Minimum interval between training cycles in seconds (default: 120)
        """
        self.logger.info(f"Starting client (duration: {duration_minutes}m, interval: {training_interval_seconds}s)")

        # Initialize tracking variables
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_training_time = 0  # Start with 0 to ensure we train immediately
        training_count = 0
        best_accuracy = 0.0

        try:
            while time.time() < end_time:
                current_time = time.time()

                # Log progress every minute
                if int(current_time) % 60 == 0:
                    time_elapsed_minutes = (current_time - start_time) / 60
                    time_remaining_minutes = duration_minutes - time_elapsed_minutes
                    self.logger.info(f"Client running: {time_elapsed_minutes:.1f}m elapsed, {time_remaining_minutes:.1f}m remaining")

                # Check if it's time to train
                time_since_last_training = current_time - last_training_time
                if time_since_last_training >= training_interval_seconds:
                    self.logger.info(f"=== Training cycle {training_count + 1} ===")

                    # Receive global model
                    global_weights = self.receive_global_model()
                    if global_weights is None:
                        self.logger.error(f"Failed to receive global model")
                        time.sleep(5)
                        continue

                    # Train local model
                    local_weights = self.train_local_model(global_weights)
                    if local_weights is None:
                        self.logger.error(f"Failed to train local model")
                        time.sleep(5)
                        continue

                    # Evaluate the model if TensorFlow is available
                    if TENSORFLOW_AVAILABLE:
                        model = create_lenet_model()
                        if model is not None:
                            model.set_weights(local_weights)
                            test_size = min(1000, len(self.X_test))
                            test_indices = np.random.choice(len(self.X_test), test_size, replace=False)
                            X_test_subset = self.X_test[test_indices]
                            y_test_subset = self.y_test[test_indices]
                            _, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)

                            # Track best accuracy
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                self.logger.info(f"New best accuracy: {best_accuracy:.4f}")

                    # Send model update
                    if not self.send_model_update(local_weights):
                        self.logger.error(f"Failed to send model update")
                        time.sleep(5)
                        continue

                    # Update tracking variables
                    last_training_time = current_time
                    training_count += 1

                    # Run garbage collection after training
                    gc.collect()

                # Sleep briefly to avoid tight polling
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("Client interrupted by user")

        # Final statistics
        self.logger.info(f"Federated learning completed: {training_count} training cycles")
        if self.best_accuracy > 0:
            self.logger.info(f"Best accuracy achieved: {self.best_accuracy:.4f}")

        # Generate visualizations
        self.generate_visualizations()

    def generate_visualizations(self):
        """Generate and save visualizations of client metrics"""
        self.logger.info("Generating visualizations")

        # Only generate visualizations if we have metrics
        if not self.metrics['train_accuracy'] and not self.metrics['test_accuracy']:
            self.logger.warning("No metrics available for visualization")
            return

        try:
            # Plot accuracy metrics
            accuracy_plot_path = plot_client_accuracy(
                {
                    'train': self.metrics['train_accuracy'],
                    'test': self.metrics['test_accuracy']
                },
                self.client_id,
                self.logger
            )
            self.logger.info(f"Accuracy plot saved to {accuracy_plot_path}")

            # Plot loss metrics
            if self.metrics['train_loss']:
                loss_plot_path = plot_client_loss(
                    self.metrics['train_loss'],
                    self.client_id,
                    self.logger
                )
                self.logger.info(f"Loss plot saved to {loss_plot_path}")

        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

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

def signal_handler(*_):
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
    duration_minutes = int(os.environ.get("DURATION_MINUTES", "60"))
    training_interval = int(os.environ.get("TRAINING_INTERVAL_SECONDS", "120"))

    # Create client
    client = FederatedClient(
        bootstrap_servers=bootstrap_servers,
        update_topic="model_updates",
        model_topic="global_model",
        client_id=client_id
    )

    # Store client instance for signal handling
    client_instance = client

    try:
        # Start federated learning
        client.start(
            duration_minutes=duration_minutes,
            training_interval_seconds=training_interval
        )
    finally:
        # Close resources
        client.close()
