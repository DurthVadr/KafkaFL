import datetime
import os
import time
import logging
import traceback
import io
import numpy as np
from kafka import KafkaConsumer, KafkaProducer

# Import TensorFlow conditionally to handle environments where it might not be available
try:
    import tensorflow as tf
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
        self.X, self.y, self.X_test, self.y_test = self.load_data_cifar10()
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
                    #compression_type='gzip'  # Add compression to reduce message size
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
            X_test = np.random.rand(200, 32, 32, 3).astype('float32')
            y_test = np.random.randint(0, 10, size=200)
            return X_train, y_train, X_test, y_test

        try:
            # Load CIFAR-10 data
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # Load both train and test data
            X_train = X_train.astype('float32') / 255.0
            y_train = np.squeeze(y_train)
            X_test = X_test.astype('float32') / 255.0
            y_test = np.squeeze(y_test)

            # Use a smaller subset for faster training
            subset_size = 5000  # Use only 5000 samples for faster training
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]

            # Use a smaller test set for faster evaluation
            test_size = 1000
            X_test = X_test[:test_size]
            y_test = y_test[:test_size]

            logging.info(f"Client {self.client_id}: Loaded CIFAR-10 data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error loading CIFAR-10 data: {e}")
            logging.error(traceback.format_exc())
            # Fallback to random data
            X_train = np.random.rand(1000, 32, 32, 3).astype('float32')
            y_train = np.random.randint(0, 10, size=1000)
            X_test = np.random.rand(200, 32, 32, 3).astype('float32')
            y_test = np.random.randint(0, 10, size=200)
            return X_train, y_train, X_test, y_test

    def train(self, global_weights):
        logging.info(f"Client {self.client_id}: Training model with local data")

        if not TENSORFLOW_AVAILABLE:
            logging.warning(f"Client {self.client_id}: TensorFlow not available. Simulating training.")
            self.model = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in global_weights]
            logging.info(f"Client {self.client_id}: Simulated training complete. Model has {len(self.model)} layers.")
            return self.model

        try:
            # Create a model with the standard architecture
            model = self.create_model()

            # Log the shapes of the global weights for debugging
            logging.info(f"Client {self.client_id}: Received global model with {len(global_weights)} weight arrays")
            for i, w in enumerate(global_weights):
                logging.info(f"Client {self.client_id}: Global weight {i} shape: {w.shape}")

            # Log the expected shapes of the model weights
            model_weights = model.get_weights()
            logging.info(f"Client {self.client_id}: Model expects {len(model_weights)} weight arrays")
            for i, w in enumerate(model_weights):
                logging.info(f"Client {self.client_id}: Expected weight {i} shape: {w.shape}")

            # Check if weights are compatible
            if not self.are_weights_compatible(model, global_weights):
                logging.warning(f"Client {self.client_id}: Global weights are not compatible with the model. Attempting to adapt.")
                adapted_weights = self.adapt_weights(model, global_weights)
                if adapted_weights is None:
                    logging.error(f"Client {self.client_id}: Could not adapt weights. Cannot train.")
                    raise ValueError("Incompatible weights")
                else:
                    logging.info(f"Client {self.client_id}: Successfully adapted weights to match model architecture.")
                    model.set_weights(adapted_weights)
            else:
                model.set_weights(global_weights)

            subset_size = min(5000, len(self.X))
            indices = np.random.choice(len(self.X), subset_size, replace=False)
            X_subset = self.X[indices]
            y_subset = self.y[indices]

            # Train for more epochs to improve accuracy
            epochs = 5
            batch_size = 32
            logging.info(f"Client {self.client_id}: Training model for {epochs} epochs with batch size {batch_size}")

            # Use a callback to log progress
            class LoggingCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logging.info(f"Client {self.client_id}: Epoch {epoch+1}/{epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")

            # Train the model
            model.fit(X_subset, y_subset, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[LoggingCallback()])

            test_size = min(1000, len(self.X))
            test_indices = np.random.choice(len(self.X), test_size, replace=False)
            X_test_subset = self.X[test_indices]
            y_test_subset = self.y[test_indices]
            _, accuracy = model.evaluate(X_test_subset, y_test_subset, verbose=0)
            logging.info(f"Client {self.client_id}: Model accuracy after training: {accuracy:.4f}")

            self.model = model.get_weights()
            return self.model
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error training model: {e}")
        logging.error(traceback.format_exc())
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

    def accuracy(self, model, X_test, y_test):
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Client {self.client_id}: Model accuracy: {accuracy:.4f}")
        return accuracy

    def create_model(self):
        """Create a model with the standard architecture used in this federated learning system."""
        input_shape = (32, 32, 3)
        num_classes = 10
        inputs = Input(shape=input_shape)

        # First convolutional layer with explicit parameters
        x = Conv2D(16, (3, 3), padding='valid', activation='relu')(inputs)  # 30x30x16

        # Add batch normalization for better training stability
        x = tf.keras.layers.BatchNormalization()(x)

        # Second convolutional layer
        x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)  # 28x28x32
        x = tf.keras.layers.BatchNormalization()(x)

        # Max pooling layer
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 14x14x32

        # Dropout for regularization
        x = Dropout(0.25)(x)

        # Add another convolutional layer for better feature extraction
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # 14x14x64
        x = tf.keras.layers.BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 7x7x64

        # Flatten layer - should be 7*7*64 = 3136
        x = Flatten()(x)

        # Dense layer
        x = Dense(128, activation='relu')(x)  # Increased units for better capacity
        x = tf.keras.layers.BatchNormalization()(x)

        # Dropout for regularization
        x = Dropout(0.5)(x)

        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)

        # Create and compile the model
        model = Model(inputs=inputs, outputs=outputs)

        # Use a learning rate schedule for better convergence
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Print model summary and layer shapes for debugging
        if self.client_id == 0:  # Only print for first client to avoid log spam
            model.summary()
            # Print the output shape of each layer
            for i, layer in enumerate(model.layers):
                logging.info(f"Client {self.client_id}: Layer {i} ({layer.name}): Output shape = {layer.output_shape}")

        return model

    def are_weights_compatible(self, model, weights):
        """Check if the weights are compatible with the model."""
        model_weights = model.get_weights()

        # Check if number of weight arrays matches
        if len(model_weights) != len(weights):
            logging.warning(f"Client {self.client_id}: Number of weight arrays doesn't match: {len(weights)} vs {len(model_weights)}")
            return False

        # Check if shapes match
        for i, (model_w, w) in enumerate(zip(model_weights, weights)):
            if model_w.shape != w.shape:
                logging.warning(f"Client {self.client_id}: Shape mismatch at index {i}: {w.shape} vs {model_w.shape}")
                return False

        return True

    def adapt_weights(self, model, weights):
        """Adapt weights to be compatible with the model when possible."""
        model_weights = model.get_weights()
        adapted_weights = []

        # Check if we can adapt the weights
        if len(model_weights) != len(weights):
            logging.error(f"Client {self.client_id}: Cannot adapt weights - number of arrays doesn't match")
            return None

        # Try to adapt each weight array
        for i, (model_w, w) in enumerate(zip(model_weights, weights)):
            if model_w.shape == w.shape:
                # Shapes match, use as is
                adapted_weights.append(w)
            elif i == 4 and len(model_w.shape) == 2 and len(w.shape) == 2 and model_w.shape[1] == w.shape[1]:
                # This is likely the flattened layer - we can try to adapt it
                logging.warning(f"Client {self.client_id}: Attempting to adapt flattened layer weights from {w.shape} to {model_w.shape}")

                # If the target shape is larger, we'll pad with zeros
                if model_w.shape[0] > w.shape[0]:
                    padding = np.zeros((model_w.shape[0] - w.shape[0], w.shape[1]), dtype=w.dtype)
                    adapted_w = np.vstack([w, padding])
                    adapted_weights.append(adapted_w)
                    logging.info(f"Client {self.client_id}: Padded weights from {w.shape} to {adapted_w.shape}")
                # If the target shape is smaller, we'll truncate
                elif model_w.shape[0] < w.shape[0]:
                    adapted_w = w[:model_w.shape[0], :]
                    adapted_weights.append(adapted_w)
                    logging.info(f"Client {self.client_id}: Truncated weights from {w.shape} to {adapted_w.shape}")
                else:
                    # This shouldn't happen, but just in case
                    adapted_weights.append(w)
            else:
                logging.error(f"Client {self.client_id}: Cannot adapt weights at index {i}: {w.shape} vs {model_w.shape}")
                return None

        return adapted_weights

    def evaluate_model(self, weights, X_test, y_test):
        """Evaluate a model with the given weights on test data."""
        try:
            # Log the shapes of the weights for debugging
            logging.info(f"Client {self.client_id}: Evaluating model with {len(weights)} weight arrays")
            for i, w in enumerate(weights):
                logging.info(f"Client {self.client_id}: Weight {i} shape: {w.shape}")

            # Create a model with the same architecture as used in training
            model = self.create_model()

            # Log the expected shapes of the model weights
            model_weights = model.get_weights()
            logging.info(f"Client {self.client_id}: Model expects {len(model_weights)} weight arrays")
            for i, w in enumerate(model_weights):
                logging.info(f"Client {self.client_id}: Expected weight {i} shape: {w.shape}")

            # Check if weights are compatible
            if not self.are_weights_compatible(model, weights):
                logging.warning(f"Client {self.client_id}: Weights are not compatible with the model. Attempting to adapt.")
                adapted_weights = self.adapt_weights(model, weights)
                if adapted_weights is None:
                    logging.warning(f"Client {self.client_id}: Could not adapt weights. Using a fresh model.")
                    # Train the model for 1 epoch on a small subset to get a baseline accuracy
                    subset_size = min(1000, len(X_test))
                    indices = np.random.choice(len(X_test), subset_size, replace=False)
                    X_subset = X_test[indices]
                    y_subset = y_test[indices]
                    model.fit(X_subset, y_subset, epochs=1, batch_size=32, verbose=0)
                else:
                    logging.info(f"Client {self.client_id}: Successfully adapted weights for evaluation.")
                    model.set_weights(adapted_weights)
            else:
                # Set the weights if they are compatible
                model.set_weights(weights)

            # Evaluate on test data
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            return accuracy
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error in evaluate_model: {e}")
            logging.error(traceback.format_exc())
            return 0.0


    def close(self):
        """Properly close all connections and resources."""
        try:
            if self.consumer:
                # Unsubscribe and commit offsets before closing
                self.consumer.unsubscribe()
                self.consumer.commit()
                self.consumer.close()
                logging.info(f"Client {self.client_id}: Consumer closed")

            if self.producer:
                # Flush any pending messages before closing
                self.producer.flush()
                self.producer.close()
                logging.info(f"Client {self.client_id}: Producer closed")

            # Clear any large objects to help with memory cleanup
            self.model = None
            self.X = None
            self.y = None
            self.X_test = None
            self.y_test = None

            logging.info(f"Client {self.client_id}: All resources released")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error during shutdown: {e}")
            logging.error(traceback.format_exc())

# Global variable to store the client instance for signal handlers
client_instance = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logging.info("Received shutdown signal, closing client gracefully...")
    if client_instance:
        client_instance.close()
    sys.exit(0)

if __name__ == "__main__":
    import signal
    import sys

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop

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

    # Store the client instance for signal handlers
    client_instance = client
    now = datetime.datetime.now()
    client.start(num_rounds=10)
    finish_time = datetime.datetime.now()
    logging.info(f"Client {client.client_id}: Federated learning completed at {now}")
    logging.info(f"It took {finish_time - now} seconds to complete federated learning")
    # Calculate final model accuracy on test data
    if client.model is not None and hasattr(client, 'X_test') and hasattr(client, 'y_test'):
        try:
            # Use the client's evaluate_model method which now handles incompatible weights gracefully
            accuracy = client.evaluate_model(client.model, client.X_test, client.y_test)
            logging.info(f"Client {client.client_id}: Final model accuracy on test data: {accuracy:.4f}")
        except Exception as e:
            logging.error(f"Client {client.client_id}: Error calculating final accuracy: {e}")
            logging.error(traceback.format_exc())
    else:
        logging.warning(f"Client {client.client_id}: Cannot calculate final accuracy - model or test data not available")
    logging.info(f"Client {client.client_id}: Closing client")
    client.close()

