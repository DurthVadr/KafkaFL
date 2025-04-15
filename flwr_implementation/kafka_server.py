import io
import time
import logging
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy

class KafkaServer:
    """Flower server that communicates with clients via Kafka."""

    def __init__(self, broker: str, strategy: Strategy, num_rounds: int = 3):
        self.broker = broker
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.model_topic = 'model_topic'  # Topic for server to send models
        self.update_topic = 'update_topic'  # Topic for clients to send updates

        # Initialize Kafka producer and consumer
        self.producer = None
        self.consumer = None
        self.client_id_counter = 0

    def connect_kafka(self):
        """Connect to Kafka broker."""
        max_attempts = 10
        attempt = 0
        initial_delay = 5
        retry_delay = 5

        logging.info(f"Waiting {initial_delay} seconds before first Kafka connection attempt...")
        time.sleep(initial_delay)

        while attempt < max_attempts:
            try:
                # Create Kafka consumer
                self.consumer = KafkaConsumer(
                    self.update_topic,  # Subscribe to update topic
                    bootstrap_servers=self.broker,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    group_id='federated_server',
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
                logging.info(f"Server: Connected to Kafka. Available topics: {topics}")

                # Subscribe to update topic
                self.consumer.subscribe([self.update_topic])
                logging.info(f"Server: Subscribed to topic: {self.update_topic}")

                return True
            except Exception as e:
                logging.error(f"Server: Failed to connect to Kafka (attempt {attempt+1}/{max_attempts}): {e}")
                attempt += 1
                if attempt < max_attempts:
                    logging.info(f"Server: Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logging.error(f"Server: Failed to connect to Kafka after {max_attempts} attempts")
        logging.error(f"Server: Please ensure that Kafka is running at {self.broker} and the topics '{self.model_topic}' and '{self.update_topic}' exist")
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

            logging.info(f"Server: Serialized weights with {len(weights)} layers, size: {len(serialized_weights)} bytes")
            return serialized_weights
        except Exception as e:
            logging.error(f"Server: Error serializing weights: {e}")
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
            logging.info(f"Server: Deserialized weights with {len(weights)} layers")
            return weights
        except Exception as e:
            logging.error(f"Server: Error deserializing weights: {e}")
            return None

    def send_model(self, parameters):
        """Send model to clients via Kafka."""
        try:
            # Convert parameters to weights
            weights = parameters_to_ndarrays(parameters)

            # Serialize the weights
            serialized_weights = self.serialize_weights(weights)
            if serialized_weights is None:
                logging.error("Server: Failed to serialize weights")
                return False

            # Send the serialized weights
            future = self.producer.send(self.model_topic, serialized_weights)
            record_metadata = future.get(timeout=10)
            self.producer.flush()

            logging.info(f"Server: Model sent to clients. Model has {len(weights)} layers.")
            logging.info(f"Server: Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Server: Error sending model: {e}")
            return False

    def receive_updates(self, min_clients, timeout_seconds=300):
        """Receive model updates from clients via Kafka."""
        try:
            client_updates = []
            clients_this_round = 0

            # Wait for updates from clients with a timeout
            start_time = time.time()
            poll_count = 0

            logging.info(f"Server: Waiting for client updates (timeout: {timeout_seconds} seconds)")

            # Clear any old messages
            self.consumer.poll(0)

            while clients_this_round < min_clients and (time.time() - start_time) < timeout_seconds:
                # Poll for messages with a timeout
                poll_result = self.consumer.poll(timeout_ms=5000)
                poll_count += 1

                if poll_result:
                    logging.info(f"Server: Poll attempt {poll_count}, received {sum(len(msgs) for msgs in poll_result.values())} messages")

                    for _, messages in poll_result.items():
                        for message in messages:
                            try:
                                # Deserialize the client update
                                weights = self.deserialize_weights(message.value)
                                if weights is None:
                                    logging.warning("Server: Failed to deserialize client update, skipping")
                                    continue

                                # Convert weights to parameters
                                parameters = ndarrays_to_parameters(weights)

                                # Assign client ID
                                client_id = self.client_id_counter
                                self.client_id_counter += 1

                                # Create fit result
                                fit_res = fl.common.FitRes(
                                    parameters=parameters,
                                    num_examples=1,  # We don't know the actual number
                                    metrics={}
                                )

                                client_updates.append((client_id, fit_res))
                                clients_this_round += 1
                                logging.info(f"Server: Received update from client {client_id} (Client {clients_this_round}/{min_clients})")

                                if clients_this_round >= min_clients:
                                    break
                            except Exception as e:
                                logging.error(f"Server: Error processing client update: {e}")

                        if clients_this_round >= min_clients:
                            break
                else:
                    logging.info(f"Server: Poll attempt {poll_count}: No messages received")
                    # Sleep a bit to avoid tight polling
                    time.sleep(1)

            logging.info(f"Server: Collected {clients_this_round} client updates after {poll_count} poll attempts")
            return client_updates
        except Exception as e:
            logging.error(f"Server: Error receiving updates: {e}")
            return []

    def run(self):
        """Run the federated learning server."""
        try:
            if not self.connect_kafka():
                logging.error("Server: Failed to connect to Kafka, exiting")
                logging.error("Server: Make sure Kafka is running and accessible before starting the server")
                return

            # Initialize parameters
            parameters = self.strategy.initialize_parameters()
            if parameters is None:
                # Initialize with random parameters if strategy doesn't provide them
                logging.info("Server: Strategy did not initialize parameters, using random initialization")
                from flwr_implementation import model as model_module
                model = model_module.create_keras_model()
                weights = model.get_weights()
                parameters = ndarrays_to_parameters(weights)

            # Send initial model to clients
            logging.info("Server: Sending initial model to clients")
            if not self.send_model(parameters):
                logging.error("Server: Failed to send initial model to clients, exiting")
                return

            # Main federated learning loop
            for round in range(self.num_rounds):
                logging.info(f"Server: Starting round {round + 1}/{self.num_rounds}")

                # Receive updates from clients
                min_clients = max(self.strategy.min_fit_clients, 1)
                client_updates = self.receive_updates(min_clients)

                if not client_updates:
                    logging.warning(f"Server: No client updates received in round {round + 1}, skipping")
                    continue

                # Aggregate updates using the strategy
                logging.info(f"Server: Aggregating {len(client_updates)} client updates")

                # Convert client_updates to the format expected by the strategy
                results = [(cid, fit_res.parameters, fit_res.num_examples) for cid, fit_res in client_updates]

                # Aggregate parameters
                parameters_aggregated = self.strategy.aggregate_fit(parameters, results)

                # Send updated model to clients
                logging.info("Server: Sending updated model to clients")
                if not self.send_model(parameters_aggregated):
                    logging.error(f"Server: Failed to send updated model in round {round + 1}")
                    continue

                # Update parameters for next round
                parameters = parameters_aggregated
                logging.info(f"Server: Round {round + 1} completed")

            logging.info(f"Server: Federated learning completed after {self.num_rounds} rounds")
        except Exception as e:
            logging.error(f"Server: Error in run loop: {e}")
        finally:
            # Close Kafka connections
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()


def start_kafka_server(broker: str, strategy: Strategy, num_rounds: int = 3):
    """Start a Flower server that connects to clients using Kafka."""
    server = KafkaServer(broker, strategy, num_rounds)
    server.run()
