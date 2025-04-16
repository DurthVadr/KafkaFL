import os
import time
import logging
import traceback
import io
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# Import TensorFlow conditionally to handle environments where it might not be available
try:
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available. Using simplified data and model.")
    TENSORFLOW_AVAILABLE = False

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
        max_attempts = 5
        attempt = 0
        retry_delay = 5

        while attempt < max_attempts:
            try:
                # Create Kafka consumer with appropriate configuration and larger message size
                self.consumer = KafkaConsumer(
                    bootstrap_servers=self.bootstrap_servers,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id=f'federated_client_{self.client_id}',
                    value_deserializer=lambda m: m,  # Keep raw bytes for now
                    max_partition_fetch_bytes=10485760,  # 10MB max fetch size
                    fetch_max_bytes=52428800  # 50MB max fetch bytes
                )

                # Create Kafka producer with larger message size and compression
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: v,  # Keep raw bytes for now
                    max_request_size=20971520,  # 20MB max message size (increased)
                    buffer_memory=41943040,  # 40MB buffer memory (increased)
                    compression_type='gzip'  # Add compression to reduce message size
                )

                # Test connection by listing topics
                topics = self.consumer.topics()
                logging.info(f"Client {self.client_id}: Available Kafka topics: {topics}")

                # Subscribe to model topic
                self.consumer.subscribe([self.model_topic])
                logging.info(f"Client {self.client_id}: Subscribed to topic: {self.model_topic}")

                logging.info(f"Client {self.client_id}: Successfully connected to Kafka")
                return True
            except Exception as e:
                logging.error(f"Client {self.client_id}: Failed to connect to Kafka (attempt {attempt+1}/{max_attempts}): {e}")
                attempt += 1
                if attempt < max_attempts:
                    logging.info(f"Client {self.client_id}: Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        return False

    def load_data_cifar10(self):
        if not TENSORFLOW_AVAILABLE:
            logging.warning("TensorFlow not available. Using random data instead of CIFAR-10.")
            # Create random data with the same shape as CIFAR-10
            X_train = np.random.rand(1000, 32, 32, 3).astype('float32')
            y_train = np.random.randint(0, 10, size=1000)
            return X_train, y_train

        try:
            # Load CIFAR-10 data
            (X_train, y_train), _ = cifar10.load_data()  # Ignore test data
            X_train = X_train.astype('float32') / 255.0
            y_train = np.squeeze(y_train)

            # Use a smaller subset for faster training
            subset_size = 5000  # Use only 5000 samples for faster training
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]

            logging.info(f"Client {self.client_id}: Loaded CIFAR-10 data with {X_train.shape[0]} samples")
            return X_train, y_train
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error loading CIFAR-10 data: {e}")
            logging.error(traceback.format_exc())
            # Fallback to random data
            X_train = np.random.rand(1000, 32, 32, 3).astype('float32')
            y_train = np.random.randint(0, 10, size=1000)
            return X_train, y_train

    def train(self, global_weights):
        logging.info(f"Client {self.client_id}: Training model with local data")

        if not TENSORFLOW_AVAILABLE:
            logging.warning(f"Client {self.client_id}: TensorFlow not available. Simulating training.")
            # Simulate training by adding random noise to the global model
            self.model = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in global_weights]
            logging.info(f"Client {self.client_id}: Simulated training complete. Model has {len(self.model)} layers.")
            return self.model

        try:
            # Initialize the model architecture with smaller dimensions
            # to match the server's model and fit within Kafka message size limits
            input_shape = (32, 32, 3)
            num_classes = 10
            inputs = Input(shape=input_shape)
            x = Conv2D(16, (3, 3), activation='relu')(inputs)  # 16 filters instead of 32
            x = Conv2D(32, (3, 3), activation='relu')(x)      # 32 filters instead of 64
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(64, activation='relu')(x)               # 64 units instead of 128
            x = Dropout(0.5)(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Set the model weights to the global weights
            model.set_weights(global_weights)

            # Train the model for a few epochs on a subset of the data
            # Use a small subset for faster training
            subset_size = min(5000, len(self.X))  # Use at most 5000 samples for training
            indices = np.random.choice(len(self.X), subset_size, replace=False)
            X_subset = self.X[indices]
            y_subset = self.y[indices]

            # Train for a small number of epochs
            model.fit(X_subset, y_subset, epochs=1, batch_size=32, verbose=0)

            # Evaluate the model
            test_size = min(1000, len(self.X))
            test_indices = np.random.choice(len(self.X), test_size, replace=False)
            X_test_subset = self.X[test_indices]
            y_test_subset = self.y[test_indices]
            _, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)
            logging.info(f"Client {self.client_id}: Model accuracy after training: {accuracy:.4f}")

            # Get the updated weights
            self.model = model.get_weights()
            return self.model
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error training model: {e}")
            logging.error(traceback.format_exc())
            # Fallback to simulating training
            self.model = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in global_weights]
            return self.model

    def serialize_model_weights(self, weights):
        try:
            # Use a more efficient serialization method
            # Create a BytesIO buffer
            buffer = io.BytesIO()

            # Save the number of arrays
            buffer.write(np.array([len(weights)], dtype=np.int32).tobytes())

            # Save each array with its shape information
            for arr in weights:
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
            logging.info(f"Client {self.client_id}: Serialized model weights with {len(weights)} layers, size: {len(serialized_weights)} bytes")
            return serialized_weights
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error serializing model weights: {e}")
            logging.error(traceback.format_exc())
            return None

    def send_update(self):
        try:
            # Serialize the model weights
            serialized_weights = self.serialize_model_weights(self.model)
            if serialized_weights is None:
                logging.error(f"Client {self.client_id}: Failed to serialize model weights")
                return False

            # Send the serialized weights
            future = self.producer.send(self.update_topic, serialized_weights)
            record_metadata = future.get(timeout=10)
            self.producer.flush()
            logging.info(f"Client {self.client_id}: Model update sent to server. Model has {len(self.model)} layers.")
            logging.info(f"Client {self.client_id}: Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending update: {e}")
            # Log the full traceback
            logging.error(traceback.format_exc())
            return False

    def deserialize_model_weights(self, buffer):
        try:
            # Deserialize using the same format as serialization
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
            logging.info(f"Client {self.client_id}: Deserialized model with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error deserializing model weights: {e}")
            logging.error(traceback.format_exc())
            return None

    def consume_model_from_topic(self):
        try:
            # Make sure we're subscribed to the model topic
            self.consumer.subscribe([self.model_topic])
            logging.info(f"Client {self.client_id}: Waiting for model from server on topic {self.model_topic}")

            # Set a timeout for receiving the model (60 seconds)
            timeout_ms = 60000
            start_time = time.time()
            poll_count = 0

            while (time.time() - start_time) < (timeout_ms/1000):
                # Poll for messages with a timeout
                poll_result = self.consumer.poll(timeout_ms=5000)
                poll_count += 1
                logging.info(f"Client {self.client_id}: Poll attempt {poll_count}, received {sum(len(msgs) for msgs in poll_result.values() if msgs)} messages")

                if poll_result:
                    for tp, messages in poll_result.items():
                        logging.info(f"Client {self.client_id}: Processing messages from topic-partition: {tp.topic}-{tp.partition}")
                        for message in messages:
                            try:
                                # Deserialize the model weights
                                model_weights = self.deserialize_model_weights(message.value)
                                if model_weights is not None:
                                    self.model = model_weights
                                    logging.info(f"Client {self.client_id}: Model received from server. Model has {len(model_weights)} layers.")
                                    logging.info(f"Client {self.client_id}: Message offset: {message.offset}, partition: {message.partition}")
                                    return model_weights  # Return the model after receiving it
                            except Exception as e:
                                logging.error(f"Client {self.client_id}: Error processing message: {e}")
                                logging.error(traceback.format_exc())
                else:
                    logging.info(f"Client {self.client_id}: No messages received in this poll")
                    # Sleep a bit to avoid tight polling
                    time.sleep(1)

            logging.error(f"Client {self.client_id}: Timeout waiting for model from server after {poll_count} poll attempts")
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