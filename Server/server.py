import os
import time
import logging
import traceback
import io
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# Import TensorFlow conditionally to handle environments where it might not be available
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available. Using simplified model initialization.")
    TENSORFLOW_AVAILABLE = False

class FederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.producer = None
        self.model_topic = model_topic
        self.update_topic = update_topic
        self.global_model = self.initialize_random_global_model()  # Use random model for testing
        self.num_rounds = 10
        self.client_id_counter = 0  # Counter for client IDs

    def connect_kafka(self):
        max_attempts = 10
        attempt = 0
        initial_delay = 15  # Wait longer initially
        retry_delay = 10
        logging.info(f"Waiting {initial_delay} seconds before first Kafka connection attempt...")
        time.sleep(initial_delay)
        while attempt < max_attempts:
            try:
                # Create Kafka consumer with appropriate deserializer and larger message size
                self.consumer = KafkaConsumer(
                    bootstrap_servers=self.bootstrap_servers,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id='federated_server',
                    value_deserializer=lambda m: m,  # Keep raw bytes for now
                    max_partition_fetch_bytes=10485760,  # 10MB max fetch size
                    fetch_max_bytes=52428800  # 50MB max fetch bytes
                )

                # Create Kafka producer with appropriate serializer and larger message size
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: v,  # Keep raw bytes for now
                    max_request_size=20971520,  # 20MB max message size (increased)
                    buffer_memory=41943040,  # 40MB buffer memory (increased)
                   # compression_type='gzip'  # Add compression to reduce message size
                )

                # Test connection by listing topics
                topics = self.consumer.topics()
                logging.info(f"Available Kafka topics: {topics}")

                # Subscribe to update topic
                self.consumer.subscribe([self.update_topic])
                logging.info(f"Subscribed to topic: {self.update_topic}")

                logging.info("Successfully connected to Kafka")
                return True
            except NoBrokersAvailable as e:  # Catch NoBrokersAvailable specifically
                logging.warning(f"No brokers available (attempt {attempt + 1}/{max_attempts}): {e}")
                attempt += 1
                time.sleep(retry_delay)  # Wait longer before retrying
            except KafkaError as e:
                logging.warning(f"Failed to connect to Kafka (attempt {attempt + 1}/{max_attempts}): {e}")
                attempt += 1
                time.sleep(retry_delay)  # Wait longer before retrying
        logging.error("Failed to connect to Kafka after multiple attempts")
        return False

    def initialize_global_model_cifar10(self):
        # Initialize global model for CIFAR-10 dataset
        if not TENSORFLOW_AVAILABLE:
            logging.warning("TensorFlow not available. Using random weights instead.")
            return self.initialize_random_global_model()

        try:
            # Create a smaller model architecture that will still be compatible with clients
            # but with fewer parameters to fit within Kafka message size limits
            # Create a model with explicit parameters to ensure consistency with clients
            input_shape = (32, 32, 3)
            num_classes = 10
            inputs = Input(shape=input_shape)

            # First convolutional layer with explicit parameters
            x = Conv2D(16, (3, 3), padding='valid', activation='relu')(inputs)  # 30x30x16

            # Add batch normalization for better training stability
            x = BatchNormalization()(x)

            # Second convolutional layer
            x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)  # 28x28x32
            x = BatchNormalization()(x)

            # Max pooling layer
            x = MaxPooling2D(pool_size=(2, 2))(x)  # 14x14x32

            # Dropout for regularization
            x = Dropout(0.25)(x)

            # Add another convolutional layer for better feature extraction
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # 14x14x64
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)  # 7x7x64

            # Flatten layer - should be 7*7*64 = 3136
            x = Flatten()(x)

            # Dense layer
            x = Dense(128, activation='relu')(x)  # Increased units for better capacity
            x = BatchNormalization()(x)

            # Dropout for regularization
            x = Dropout(0.5)(x)

            # Output layer
            outputs = Dense(num_classes, activation='softmax')(x)

            # Create and compile the model
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Print model summary to verify the architecture
            model.summary()

            # Print the output shape of each layer for debugging
            for i, layer in enumerate(model.layers):
                logging.info(f"Layer {i} ({layer.name}): Output shape = {layer.output_shape}")

            weights = model.get_weights()
            logging.info(f"Global Model CIFAR-10 initialized with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Error initializing CIFAR-10 model: {e}")
            logging.error(traceback.format_exc())
            return self.initialize_random_global_model()


    def initialize_random_global_model(self):
        # Initialize a random global model that matches the client model architecture
        # but with smaller dimensions to fit within Kafka message size limits
        try:
            # Create a list of numpy arrays to simulate model weights
            # Using a smaller version of the client's model architecture
            # Create a model with the same architecture as in initialize_global_model_cifar10
            # to ensure consistent weight shapes
            input_shape = (32, 32, 3)
            num_classes = 10
            inputs = Input(shape=input_shape)

            # First convolutional layer with explicit parameters
            x = Conv2D(16, (3, 3), padding='valid', activation='relu')(inputs)  # 30x30x16

            # Add batch normalization for better training stability
            x = BatchNormalization()(x)

            # Second convolutional layer
            x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)  # 28x28x32
            x = BatchNormalization()(x)

            # Max pooling layer
            x = MaxPooling2D(pool_size=(2, 2))(x)  # 14x14x32

            # Dropout for regularization
            x = Dropout(0.25)(x)

            # Add another convolutional layer for better feature extraction
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # 14x14x64
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)  # 7x7x64

            # Flatten layer - should be 7*7*64 = 3136
            x = Flatten()(x)

            # Dense layer
            x = Dense(128, activation='relu')(x)  # Increased units for better capacity
            x = BatchNormalization()(x)

            # Dropout for regularization
            x = Dropout(0.5)(x)

            # Output layer
            outputs = Dense(num_classes, activation='softmax')(x)

            # Create the model
            model = Model(inputs=inputs, outputs=outputs)

            # Initialize with random weights
            weights = [np.random.rand(*w.shape).astype(np.float32) for w in model.get_weights()]

            # Print the shapes for debugging
            for i, w in enumerate(weights):
                logging.info(f"Random weight {i} shape: {w.shape}")
            logging.info(f"Random global model initialized with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Error initializing random model: {e}")
            logging.error(traceback.format_exc())
            # Fallback to very simple model
            return [np.random.rand(10).astype(np.float32)]


    def assign_client_id(self):
        # Assign a unique client ID
        client_id = self.client_id_counter
        self.client_id_counter += 1
        return client_id

    def federated_averaging(self, client_updates):
        # Perform federated averaging on model weights
        logging.info("Performing federated averaging")
        logging.info(f"Number of client updates received: {len(client_updates)}")

        if client_updates is None or len(client_updates) == 0:
            logging.warning("No client updates received for federated averaging")
            return self.global_model

        # For TensorFlow model weights, we need to average each layer separately
        # Each client update is a list of numpy arrays (one per layer)
        averaged_weights = []

        # Get the number of layers in the model
        num_layers = len(self.global_model)

        # For each layer in the model
        for i in range(num_layers):
            # Extract the weights for this layer from all clients
            layer_updates = [update[i] for update in client_updates]
            # Average the weights for this layer
            layer_avg = np.mean(layer_updates, axis=0)
            averaged_weights.append(layer_avg)

        self.global_model = averaged_weights
        logging.info(f"Global model updated with federated averaging. Model has {len(self.global_model)} layers.")
        return self.global_model

    def send_model(self):
        try:
            # Use a more efficient serialization method
            import io
            import numpy as np

            # Create a BytesIO buffer
            buffer = io.BytesIO()

            # Save the number of arrays
            buffer.write(np.array([len(self.global_model)], dtype=np.int32).tobytes())

            # Save each array with its shape information
            for arr in self.global_model:
                # Save shape information
                shape = np.array(arr.shape, dtype=np.int32)
                buffer.write(np.array([len(shape)], dtype=np.int32).tobytes())  # Number of dimensions
                buffer.write(shape.tobytes())  # Shape dimensions

                # Save the array data
                buffer.write(arr.tobytes())

            # Get the serialized data
            serialized_weights = buffer.getvalue()
            buffer.close()

            # Log the size of the serialized data
            logging.info(f"Serialized model size: {len(serialized_weights)} bytes")

            # Send the serialized weights
            future = self.producer.send(self.model_topic, serialized_weights)
            record_metadata = future.get(timeout=10)  # Wait for send to complete
            self.producer.flush()

            logging.info(f"Global model sent to all clients. Model has {len(self.global_model)} layers.")
            logging.info(f"Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Error sending model: {e}")
            logging.error(traceback.format_exc())
            return False

    def deserialize_client_update(self, buffer):
        try:
            # Deserialize using the same format as serialization
            import io
            import numpy as np


            buffer_io = io.BytesIO(buffer)


            num_arrays = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]


            weights = []
            for _ in range(num_arrays):

                ndim = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]  # Number of dimensions
                shape = tuple(np.frombuffer(buffer_io.read(4 * ndim), dtype=np.int32))  # Shape dimensions


                size = np.prod(shape) * 4


                arr_data = np.frombuffer(buffer_io.read(int(size)), dtype=np.float32).reshape(shape)
                weights.append(arr_data)

            buffer_io.close()
            logging.info(f"Deserialized client update with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Error deserializing client update: {e}")
            logging.error(traceback.format_exc())
            return None

    def  start(self):
        if not self.connect_kafka():
            logging.error("Failed to start server due to Kafka connection issues")
            return

        try:
            # Send initial model to clients
            logging.info("Sending initial model to clients")
            if not self.send_model():
                logging.error("Failed to send initial model to clients")
                return

            for round in range(self.num_rounds):
                logging.info(f"\n\n===== Starting Round {round + 1}/{self.num_rounds} =====\n")
                client_updates = []
                clients_this_round = 0
                max_clients_per_round = 3  # We want exactly 3 clients

                # Wait for updates from clients with a timeout
                timeout_ms = 60000 * max_clients_per_round  # 60 seconds per client
                start_time = time.time()
                poll_count = 0

                logging.info(f"Waiting for client updates (timeout: {timeout_ms/1000} seconds)")

                # Clear any old messages
                self.consumer.poll(0)

                while clients_this_round < max_clients_per_round and (time.time() - start_time) < (timeout_ms/1000):
                    # Poll for messages with a timeout
                    poll_result = self.consumer.poll(timeout_ms=5000)
                    poll_count += 1

                    if poll_result:
                        logging.info(f"Poll attempt {poll_count}, received {sum(len(msgs) for msgs in poll_result.values())} messages")

                        for tp, messages in poll_result.items():
                            logging.info(f"Processing messages from topic-partition: {tp.topic}-{tp.partition}")

                            for message in messages:
                                try:
                                    # Deserialize the client update
                                    client_update = self.deserialize_client_update(message.value)
                                    if client_update is None:
                                        logging.warning("Failed to deserialize client update, skipping")
                                        continue

                                    client_id = self.assign_client_id()  # Assign client ID upon receiving update
                                    client_updates.append(client_update)
                                    clients_this_round += 1
                                    logging.info(f"Server: Received update from client {client_id} (Client {clients_this_round}/{max_clients_per_round})")

                                    if clients_this_round >= max_clients_per_round:
                                        break
                                except Exception as e:
                                    logging.error(f"Error processing client update: {e}")
                                    logging.error(traceback.format_exc())

                            if clients_this_round >= max_clients_per_round:
                                break
                    else:
                        logging.info(f"Poll attempt {poll_count}: No messages received")
                        # Sleep a bit to avoid tight polling
                        time.sleep(1)

                logging.info(f"Collected {clients_this_round} client updates for round {round + 1} after {poll_count} poll attempts")

                if clients_this_round > 0:
                    # Update global model
                    self.global_model = self.federated_averaging(client_updates)
                    # Send updated model to clients
                    if self.send_model():
                        logging.info(f"Round {round + 1} completed. Updated model sent to clients.")
                    else:
                        logging.error(f"Failed to send updated model to clients in round {round + 1}")
                else:
                    logging.warning(f"No client updates received in round {round + 1}. Skipping model update.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            logging.info("Closing Kafka connections")
            self.consumer.close()  # Ensure the consumer is closed properly
            self.producer.close()


    def close(self):
        """Properly close all connections and resources."""
        try:
            logging.info("Closing Kafka connections")

            if hasattr(self, 'consumer') and self.consumer:
                # Unsubscribe and commit offsets before closing
                self.consumer.unsubscribe()
                self.consumer.commit()
                self.consumer.close()
                logging.info("Consumer closed")

            if hasattr(self, 'producer') and self.producer:
                # Flush any pending messages before closing
                self.producer.flush()
                self.producer.close()
                logging.info("Producer closed")

            # Clear any large objects to help with memory cleanup
            self.global_model = None

            logging.info("Server closed - all resources released")
        except Exception as e:
            logging.error(f"Error during server shutdown: {e}")
            logging.error(traceback.format_exc())

# Global variable to store the server instance for signal handlers
server_instance = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logging.info(f"Received shutdown signal {sig}, closing server gracefully...")
    if server_instance:
        server_instance.close()
    sys.exit(0)

if __name__ == "__main__":
    import signal
    import sys

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop

    logging.basicConfig(level=logging.INFO)
    logging.info("Server started")

    # Use localhost instead of kafka hostname when running outside Docker
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    logging.info(f"Using Kafka bootstrap servers: {bootstrap_servers}")

    server = FederatedServer(
        bootstrap_servers=bootstrap_servers,
        model_topic='model_topic',
        update_topic='update_topic',
    )

    # Store the server instance for signal handlers
    server_instance = server

    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        logging.error(traceback.format_exc())
    finally:
        server.close()