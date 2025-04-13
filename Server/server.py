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
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
                    max_request_size=10485760,  # 10MB max message size
                    buffer_memory=20971520  # 20MB buffer memory
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
            input_shape = (32, 32, 3)
            num_classes = 10
            inputs = Input(shape=input_shape)
            x = Conv2D(32, (3, 3), activation='relu')(inputs)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            weights = model.get_weights()
            logging.info(f"Global Model CIFAR-10 initialized with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Error initializing CIFAR-10 model: {e}")
            logging.error(traceback.format_exc())
            return self.initialize_random_global_model()


    def initialize_random_global_model(self):
        # Initialize a very small random global model for testing
        try:
            # Create a list of numpy arrays to simulate model weights
            # Using tiny dimensions to reduce message size
            weights = [
                np.random.rand(3, 3, 3, 4).astype(np.float32),  # Tiny Conv2D weights
                np.random.rand(4).astype(np.float32),           # Conv2D bias
                np.random.rand(3, 3, 4, 8).astype(np.float32),  # Tiny Conv2D weights
                np.random.rand(8).astype(np.float32),           # Conv2D bias
                np.random.rand(32, 16).astype(np.float32),      # Tiny Dense weights
                np.random.rand(16).astype(np.float32),          # Dense bias
                np.random.rand(16, 10).astype(np.float32),      # Output layer weights
                np.random.rand(10).astype(np.float32)           # Output layer bias
            ]
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

            # Create a BytesIO buffer from the received bytes
            buffer_io = io.BytesIO(buffer)

            # Read the number of arrays
            num_arrays = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]

            # Read each array
            weights = []
            for _ in range(num_arrays):
                # Read shape information
                ndim = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]  # Number of dimensions
                shape = tuple(np.frombuffer(buffer_io.read(4 * ndim), dtype=np.int32))  # Shape dimensions

                # Calculate the size of the array in bytes
                size = np.prod(shape) * 4  # Assuming float32 (4 bytes per element)

                # Read the array data
                arr_data = np.frombuffer(buffer_io.read(int(size)), dtype=np.float32).reshape(shape)
                weights.append(arr_data)

            buffer_io.close()
            logging.info(f"Deserialized client update with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Error deserializing client update: {e}")
            logging.error(traceback.format_exc())
            return None

    def start(self):
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
if __name__ == "__main__":

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

    server.start()