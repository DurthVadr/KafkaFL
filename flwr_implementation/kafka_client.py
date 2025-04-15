import io
import time
import logging
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client.client import Client
from flwr.client.numpy_client import NumPyClient

class KafkaClientProxy:
    """Flower client that communicates with the server via Kafka."""

    def __init__(self, client: Client, broker: str, client_id: str):
        self.client = client
        self.broker = broker
        self.client_id = client_id
        self.model_topic = 'model_topic'  # Topic for server to send models
        self.update_topic = 'update_topic'  # Topic for clients to send updates
        self.client_topic = f'client_{client_id}'  # Unique topic for this client

        # Initialize Kafka producer and consumer
        self.producer = None
        self.consumer = None
        self.connect_kafka()

    def connect_kafka(self):
        """Connect to Kafka broker."""
        max_attempts = 5
        attempt = 0
        retry_delay = 5

        while attempt < max_attempts:
            try:
                # Create Kafka consumer
                self.consumer = KafkaConsumer(
                    self.model_topic,  # Subscribe to model topic
                    bootstrap_servers=self.broker,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id=f'federated_client_{self.client_id}',
                    value_deserializer=lambda m: m,  # Keep raw bytes
                    max_partition_fetch_bytes=10485760,  # 10MB max fetch size
                    fetch_max_bytes=52428800  # 50MB max fetch bytes
                )

                # Create Kafka producer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.broker,
                    value_serializer=lambda v: v,  # Keep raw bytes
                    max_request_size=10485760,  # 10MB max message size
                    buffer_memory=20971520  # 20MB buffer memory
                )

                # Test connection
                topics = self.consumer.topics()
                logging.info(f"Client {self.client_id}: Connected to Kafka. Available topics: {topics}")
                return True
            except Exception as e:
                logging.error(f"Client {self.client_id}: Failed to connect to Kafka (attempt {attempt+1}/{max_attempts}): {e}")
                attempt += 1
                if attempt < max_attempts:
                    logging.info(f"Client {self.client_id}: Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logging.error(f"Client {self.client_id}: Failed to connect to Kafka after {max_attempts} attempts")
        logging.error(f"Client {self.client_id}: Please ensure that Kafka is running at {self.broker} and the topic '{self.model_topic}' exists")
        return False

    def serialize_weights(self, weights):
        """Serialize model weights to bytes."""
        try:
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

            logging.info(f"Client {self.client_id}: Serialized weights with {len(weights)} layers, size: {len(serialized_weights)} bytes")
            return serialized_weights
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error serializing weights: {e}")
            return None

    def deserialize_weights(self, buffer):
        """Deserialize model weights from bytes."""
        try:
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
            logging.info(f"Client {self.client_id}: Deserialized weights with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error deserializing weights: {e}")
            return None

    def receive_model(self):
        """Receive model from server via Kafka."""
        try:
            logging.info(f"Client {self.client_id}: Waiting for model from server on topic {self.model_topic}")

            # Set a timeout for receiving the model (60 seconds)
            timeout_ms = 60000
            start_time = time.time()
            poll_count = 0

            while (time.time() - start_time) < (timeout_ms/1000):
                # Poll for messages with a timeout
                poll_result = self.consumer.poll(timeout_ms=5000)
                poll_count += 1

                if poll_result:
                    logging.info(f"Client {self.client_id}: Poll attempt {poll_count}, received {sum(len(msgs) for msgs in poll_result.values())} messages")

                    for _, messages in poll_result.items():
                        for message in messages:
                            try:
                                # Deserialize the model weights
                                weights = self.deserialize_weights(message.value)
                                if weights is not None:
                                    logging.info(f"Client {self.client_id}: Model received from server. Model has {len(weights)} layers.")
                                    # Convert weights to Flower parameters
                                    parameters = ndarrays_to_parameters(weights)
                                    return parameters
                            except Exception as e:
                                logging.error(f"Client {self.client_id}: Error processing message: {e}")
                else:
                    logging.info(f"Client {self.client_id}: No messages received in poll attempt {poll_count}")
                    # Sleep a bit to avoid tight polling
                    time.sleep(1)

            logging.error(f"Client {self.client_id}: Timeout waiting for model from server after {poll_count} poll attempts")
            return None
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error receiving model: {e}")
            return None

    def send_update(self, parameters):
        """Send model update to server via Kafka."""
        try:
            # Convert parameters to weights
            weights = parameters_to_ndarrays(parameters)

            # Serialize the weights
            serialized_weights = self.serialize_weights(weights)
            if serialized_weights is None:
                logging.error(f"Client {self.client_id}: Failed to serialize weights")
                return False

            # Send the serialized weights
            future = self.producer.send(self.update_topic, serialized_weights)
            record_metadata = future.get(timeout=10)
            self.producer.flush()

            logging.info(f"Client {self.client_id}: Update sent to server. Model has {len(weights)} layers.")
            logging.info(f"Client {self.client_id}: Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending update: {e}")
            return False

    def run(self):
        """Run the federated learning process."""
        try:
            # Check if Kafka connection was successful
            if self.consumer is None or self.producer is None:
                logging.error(f"Client {self.client_id}: Kafka connection failed, cannot start federated learning")
                return

            # Main federated learning loop
            while True:
                # Receive model from server
                parameters = self.receive_model()
                if parameters is None:
                    logging.error(f"Client {self.client_id}: Failed to receive model, exiting")
                    break

                # Get fit instructions from parameters
                fit_ins = fl.common.FitIns(parameters, {})

                # Perform local training
                logging.info(f"Client {self.client_id}: Training model with local data")
                fit_res = self.client.fit(fit_ins)

                # Send update to server
                if not self.send_update(fit_res.parameters):
                    logging.error(f"Client {self.client_id}: Failed to send update, exiting")
                    break

                logging.info(f"Client {self.client_id}: Round completed")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error in run loop: {e}")
        finally:
            # Close Kafka connections
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()


def start_kafka_client(broker: str, client: fl.client.Client = None, numpy_client: fl.client.NumPyClient = None, clientid: str = None):
    """Start a Flower client that connects to a server using Kafka."""
    if client is None and numpy_client is None:
        raise ValueError("Either client or numpy_client must be provided")

    if client is None and numpy_client is not None:
        client = numpy_client.to_client()

    # Generate client ID if not provided
    if clientid is None:
        clientid = str(int(time.time() * 1000))

    # Create and run Kafka client proxy
    kafka_client = KafkaClientProxy(client, broker, clientid)
    kafka_client.run()
