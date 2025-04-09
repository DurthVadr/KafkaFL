from kafka import KafkaConsumer, KafkaProducer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import logging
import time
import os
import traceback

class FederatedClient:
    _client_id_counter = 0  # Class-level counter for unique client IDs

    def __init__(self, bootstrap_servers, update_topic, model_topic, client_id=None):
        self.bootstrap_servers = bootstrap_servers
        self.update_topic = update_topic
        self.model_topic = model_topic

        # Use provided client_id if available, otherwise generate one
        if client_id is not None:
            self.client_id = int(client_id)
        else:
            self.client_id = FederatedClient.generate_client_id()  # Assign unique ID

        self.model = None
        self.X, self.y = self.load_data_cifar10()
        self.producer = None
        self.consumer = None
        self.connect_kafka()

    @classmethod
    def generate_client_id(cls):
        """Generates a unique client ID."""
        client_id = cls._client_id_counter
        cls._client_id_counter += 1
        return client_id

    def connect_kafka(self):
        try:
            self.consumer = KafkaConsumer(
                self.model_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32)
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.tobytes()
            )
            logging.info(f"Client {self.client_id}: Successfully connected to Kafka")
            return True
        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to connect to Kafka: {e}")
            return False

    def load_data_cifar10(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        #X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten images
        #X_test = X_test.reshape(X_test.shape[0], -1)
        #X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)  # Reduce data size
        return X_train, y_train

    def train(self, global_weights):
        logging.info(f"Client {self.client_id}: Training model with CIFAR-10 data")

        # Initialize the model architecture
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

        # Set the model weights to the global weights
        model.set_weights(global_weights)

        # Train the model for a few epochs on a subset of the data
        # Use a small subset for faster training
        subset_size = 5000  # Use 5000 samples for training
        indices = np.random.choice(len(self.X), subset_size, replace=False)
        X_subset = self.X[indices]
        y_subset = self.y[indices]

        # Train for a small number of epochs
        model.fit(X_subset, y_subset, epochs=1, batch_size=32, verbose=1)

        # Evaluate the model
        test_indices = np.random.choice(len(self.X), 1000, replace=False)
        X_test_subset = self.X[test_indices]
        y_test_subset = self.y[test_indices]
        _, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)
        logging.info(f"Client {self.client_id}: Model accuracy after training: {accuracy:.4f}")

        # Get the updated weights
        self.model = model.get_weights()
        return self.model

    def serialize_model_weights(self, weights):
        # Serialize the model weights
        # We need to flatten and concatenate all weights with metadata to reconstruct
        # First, calculate total size and create a buffer
        total_size = sum(w.size for w in weights)
        buffer = np.zeros(total_size + len(weights) * 2, dtype=np.float32)

        # Store metadata and weights in the buffer
        idx = 0
        for weights_array in weights:
            # Store shape information (limited to 2D for simplicity)
            if weights_array.ndim == 1:
                buffer[idx] = weights_array.shape[0]
                buffer[idx+1] = 1
                idx += 2
            else:
                buffer[idx] = weights_array.shape[0]
                buffer[idx+1] = weights_array.shape[1] if weights_array.ndim > 1 else 1
                idx += 2

            # Flatten and store the weights
            flat_weights = weights_array.flatten()
            buffer[idx:idx+flat_weights.size] = flat_weights
            idx += flat_weights.size

        return buffer

    def send_update(self):
        try:
            # Serialize the model weights
            serialized_weights = self.serialize_model_weights(self.model)

            # Send the serialized weights
            self.producer.send(self.update_topic, serialized_weights)
            self.producer.flush()
            logging.info(f"Client {self.client_id}: Model update sent to server. Buffer size: {serialized_weights.size}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending update: {e}")

    def deserialize_model_weights(self, buffer):
        # Deserialize the model weights from the buffer
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

    def consume_model_from_topic(self):
        try:
            self.consumer.subscribe([self.model_topic])
            logging.info(f"Client {self.client_id}: Subscribed to topic {self.model_topic}")

            # Set a timeout for receiving the model (30 seconds)
            timeout_ms = 30000
            start_time = time.time()

            while (time.time() - start_time) < (timeout_ms/1000):
                # Poll for messages with a timeout
                poll_result = self.consumer.poll(timeout_ms=5000)

                if poll_result:
                    for _, messages in poll_result.items():
                        for message in messages:
                            # Deserialize the model weights
                            model_weights = self.deserialize_model_weights(message.value)
                            self.model = model_weights
                            logging.info(f"Client {self.client_id}: Model received from server. Model has {len(model_weights)} layers.")
                            return model_weights  # Return the model after receiving it

            logging.error(f"Client {self.client_id}: Timeout waiting for model from server")
            return None
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error consuming model: {e}")
            logging.error(traceback.format_exc())
            return None


    def register_client(self):
        # Send a registration request to the server
        try:
            self.producer.send(self.update_topic, np.array([self.client_id], dtype=np.int32).tobytes())  # Sending client ID
            self.producer.flush()
            logging.info(f"Client {self.client_id}: Registration request sent to server")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending registration: {e}")



    def start(self, num_rounds=10):
        logging.info(f"Client {self.client_id}: Starting")
        if not self.connect_kafka():
            logging.error(f"Client {self.client_id}: Failed to connect to Kafka, exiting")
            return

        #self.register_client()  # Register the client

        for round in range(num_rounds):
            logging.info(f"Client {self.client_id}: Starting round {round + 1}")

            global_model = self.consume_model_from_topic()  # Get the global model
            if global_model is None:
                logging.error(f"Client {self.client_id}: Failed to receive model, exiting round")
                continue

            local_update = self.train(global_model)  # Train the local model
            if local_update is None:
                logging.error(f"Client {self.client_id}: Training failed, exiting round")
                continue

            self.send_update()  # Send the update to the server
            logging.info(f"Client {self.client_id}: Round {round + 1} completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    client_id = os.environ.get("CLIENT_ID", None)

    if client_id:
        logging.info(f"Starting client with ID: {client_id}")
    else:
        logging.info("No client ID provided, will generate one automatically")

    client = FederatedClient(
        bootstrap_servers=bootstrap_servers,
        update_topic='update_topic',
        model_topic='model_topic',
        client_id=client_id
    )

    client.start(num_rounds=10)