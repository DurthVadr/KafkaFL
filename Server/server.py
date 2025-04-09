import os
import time
import logging
from kafka import KafkaConsumer, KafkaProducer
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from kafka.errors import KafkaError, NoBrokersAvailable

class FederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.producer = None
        self.model_topic = model_topic
        self.update_topic = update_topic
        self.global_model = self.initialize_global_model_cifar10()
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
                self.consumer = KafkaConsumer(
                    'update_topic',
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32)
                )
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: v.tobytes()
                )
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

        logging.info("Global Model CIFAR-10 initialized. Weights: {}".format(model.get_weights()))
        return model.get_weights()


    def initialize_random_global_model(self):
        # Initialize a random global model
        global_model = np.random.rand(10)
        logging.info(f"Global model initialized with random weights: {global_model}")
        return global_model


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
        # Serialize the model weights
        # We need to flatten and concatenate all weights with metadata to reconstruct
        # First, calculate total size and create a buffer
        total_size = sum(w.size for w in self.global_model)
        buffer = np.zeros(total_size + len(self.global_model) * 2, dtype=np.float32)

        # Store metadata and weights in the buffer
        idx = 0
        for _, weights in enumerate(self.global_model):
            # Store shape information (limited to 2D for simplicity)
            if weights.ndim == 1:
                buffer[idx] = weights.shape[0]
                buffer[idx+1] = 1
                idx += 2
            else:
                buffer[idx] = weights.shape[0]
                buffer[idx+1] = weights.shape[1] if weights.ndim > 1 else 1
                idx += 2

            # Flatten and store the weights
            flat_weights = weights.flatten()
            buffer[idx:idx+flat_weights.size] = flat_weights
            idx += flat_weights.size

        # Send the serialized weights
        self.producer.send(self.model_topic, buffer)
        self.producer.flush()
        logging.info(f"Global model sent to all clients. Buffer size: {buffer.size}")

    def deserialize_client_update(self, buffer):
        # Deserialize the client update from the buffer
        # Extract metadata and reconstruct the weights
        weights = []
        idx = 0

        while idx < buffer.size:
            # Check if we have enough data for metadata
            if idx + 2 > buffer.size:
                break

            # Extract shape information
            shape_dim1 = int(buffer[idx])
            shape_dim2 = int(buffer[idx+1])
            idx += 2

            # Calculate the size of this weight matrix
            weight_size = shape_dim1 * shape_dim2

            # Check if we have enough data for the weights
            if idx + weight_size > buffer.size:
                break

            # Extract and reshape the weights
            flat_weights = buffer[idx:idx+weight_size]
            if shape_dim2 == 1:
                # 1D weights
                weight_matrix = flat_weights
            else:
                # 2D weights
                weight_matrix = flat_weights.reshape(shape_dim1, shape_dim2)

            weights.append(weight_matrix)
            idx += weight_size

        return weights

    def start(self):
        if not self.connect_kafka():
            logging.error("Failed to start server due to Kafka connection issues")
            return

        try:
            # Send initial model to clients
            logging.info("Sending initial model to clients")
            self.send_model()

            for round in range(self.num_rounds):
                logging.info(f"\n\n===== Starting Round {round + 1}/{self.num_rounds} =====\n")
                client_updates = []
                clients_this_round = 0
                max_clients_per_round = 3  # We want exactly 3 clients

                # Collect updates from clients
                self.consumer.subscribe([self.update_topic])  # Ensure the consumer is subscribed

                # Set a timeout for collecting updates (30 seconds per client)
                self.consumer.poll(0)  # Clear any old messages

                # Wait for updates from clients with a timeout
                timeout_ms = 30000 * max_clients_per_round  # 30 seconds per client
                start_time = time.time()

                while clients_this_round < max_clients_per_round and (time.time() - start_time) < (timeout_ms/1000):
                    # Poll for messages with a timeout
                    for _, messages in self.consumer.poll(timeout_ms=5000).items():
                        for message in messages:
                            # Deserialize the client update
                            client_update = self.deserialize_client_update(message.value)
                            client_id = self.assign_client_id()  # Assign client ID upon receiving update
                            client_updates.append(client_update)
                            clients_this_round += 1
                            logging.info(f"Server: Received update from client {client_id} (Client {clients_this_round}/{max_clients_per_round})")

                            if clients_this_round >= max_clients_per_round:
                                break
                        if clients_this_round >= max_clients_per_round:
                            break

                logging.info(f"Collected {clients_this_round} client updates for round {round + 1}")

                if clients_this_round > 0:
                    # Update global model
                    self.global_model = self.federated_averaging(client_updates)
                    # Send updated model to clients
                    self.send_model()
                    logging.info(f"Round {round + 1} completed. Updated model sent to clients.")
                else:
                    logging.warning(f"No client updates received in round {round + 1}. Skipping model update.")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            self.consumer.close()  # Ensure the consumer is closed properly
            self.producer.close()
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Server started")

    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9092")
    server = FederatedServer(
        bootstrap_servers=bootstrap_servers,
        model_topic='model_topic',
        update_topic='update_topic',
    )

    server.start()