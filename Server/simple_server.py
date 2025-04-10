import os
import time
import logging
from kafka import KafkaConsumer, KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import numpy as np
from kafka.errors import KafkaError, NoBrokersAvailable, TopicAlreadyExistsError
import traceback

class SimpleFederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        self.bootstrap_servers = bootstrap_servers
        self.consumer = None
        self.producer = None
        self.model_topic = model_topic
        self.update_topic = update_topic
        self.global_model = self.initialize_random_global_model()
        self.num_rounds = 10
        self.client_id_counter = 0  # Counter for client IDs

    def create_topics(self):
        """Create Kafka topics if they don't exist"""
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                client_id='admin-client'
            )

            # Create topics with 1 partition and replication factor 1
            topic_list = [
                NewTopic(name=self.model_topic, num_partitions=1, replication_factor=1),
                NewTopic(name=self.update_topic, num_partitions=1, replication_factor=1)
            ]

            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            logging.info(f"Created topics: {self.model_topic}, {self.update_topic}")
            admin_client.close()
            return True
        except TopicAlreadyExistsError:
            logging.info("Topics already exist")
            return True
        except Exception as e:
            logging.error(f"Failed to create topics: {e}")
            logging.error(traceback.format_exc())
            return False

    def connect_kafka(self):
        max_attempts = 10
        attempt = 0
        initial_delay = 15  # Wait longer initially
        retry_delay = 10
        logging.info(f"Waiting {initial_delay} seconds before first Kafka connection attempt...")
        time.sleep(initial_delay)
        while attempt < max_attempts:
            try:
                # Create topics first
                if not self.create_topics():
                    logging.warning("Failed to create topics, will try again")
                    attempt += 1
                    time.sleep(retry_delay)
                    continue

                # Create consumer
                self.consumer = KafkaConsumer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32),
                    auto_offset_reset='latest',
                    group_id='server',
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000
                )

                # Create producer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: v.tobytes(),
                    acks='all',
                    retries=5
                )

                # Test producer by sending a message to model_topic
                test_message = np.array([0.0], dtype=np.float32)
                self.producer.send(self.model_topic, test_message)
                self.producer.flush()
                logging.info(f"Test message sent to {self.model_topic}")

                logging.info(f"Successfully connected to Kafka at {self.bootstrap_servers}")
                return True
            except NoBrokersAvailable as e:  # Catch NoBrokersAvailable specifically
                logging.warning(f"No brokers available (attempt {attempt + 1}/{max_attempts}): {e}")
                logging.warning(traceback.format_exc())
                attempt += 1
                time.sleep(retry_delay)  # Wait longer before retrying
            except KafkaError as e:
                logging.warning(f"Failed to connect to Kafka (attempt {attempt + 1}/{max_attempts}): {e}")
                logging.warning(traceback.format_exc())
                attempt += 1
                time.sleep(retry_delay)  # Wait longer before retrying
            except Exception as e:
                logging.warning(f"Unexpected error connecting to Kafka (attempt {attempt + 1}/{max_attempts}): {e}")
                logging.warning(traceback.format_exc())
                attempt += 1
                time.sleep(retry_delay)
        logging.error("Failed to connect to Kafka after multiple attempts")
        return False

    def initialize_random_global_model(self):
        # Initialize a random global model (100 parameters)
        global_model = np.random.rand(100).astype(np.float32)
        logging.info(f"Global model initialized with random weights. Size: {global_model.size}")
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

        # Average the client updates
        averaged_weights = np.mean(client_updates, axis=0)
        self.global_model = averaged_weights
        logging.info(f"Global model updated with federated averaging. Size: {self.global_model.size}")
        return self.global_model

    def send_model(self):
        # Send the global model
        try:
            future = self.producer.send(self.model_topic, self.global_model)
            record_metadata = future.get(timeout=10)
            self.producer.flush()
            logging.info(f"Global model sent to all clients. Size: {self.global_model.size}")
            logging.info(f"Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Error sending model: {e}")
            logging.error(traceback.format_exc())
            return False

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

                # Collect updates from clients
                try:
                    self.consumer.subscribe([self.update_topic])  # Ensure the consumer is subscribed
                    logging.info(f"Subscribed to topic: {self.update_topic}")

                    # Set a timeout for collecting updates (30 seconds per client)
                    poll_result = self.consumer.poll(0)  # Clear any old messages
                    logging.info(f"Cleared old messages: {poll_result}")

                    # Wait for updates from clients with a timeout
                    timeout_ms = 30000 * max_clients_per_round  # 30 seconds per client
                    start_time = time.time()

                    while clients_this_round < max_clients_per_round and (time.time() - start_time) < (timeout_ms/1000):
                        # Poll for messages with a timeout
                        poll_result = self.consumer.poll(timeout_ms=5000)
                        logging.info(f"Poll result: {poll_result}, keys: {list(poll_result.keys())}")

                        for tp, messages in poll_result.items():
                            logging.info(f"Processing messages from topic-partition: {tp.topic}-{tp.partition}")
                            for message in messages:
                                try:
                                    # Get the client update
                                    client_update = message.value
                                    client_id = self.assign_client_id()  # Assign client ID upon receiving update
                                    client_updates.append(client_update)
                                    clients_this_round += 1
                                    logging.info(f"Server: Received update from client {client_id} (Client {clients_this_round}/{max_clients_per_round})")
                                    logging.info(f"Message offset: {message.offset}, partition: {message.partition}")

                                    if clients_this_round >= max_clients_per_round:
                                        break
                                except Exception as e:
                                    logging.error(f"Error processing message: {e}")
                                    logging.error(traceback.format_exc())
                            if clients_this_round >= max_clients_per_round:
                                break
                except Exception as e:
                    logging.error(f"Error collecting client updates: {e}")
                    logging.error(traceback.format_exc())

                logging.info(f"Collected {clients_this_round} client updates for round {round + 1}")

                if clients_this_round > 0:
                    # Update global model
                    self.global_model = self.federated_averaging(client_updates)
                    # Send updated model to clients
                    if self.send_model():
                        logging.info(f"Round {round + 1} completed. Updated model sent to clients.")
                    else:
                        logging.error(f"Failed to send updated model in round {round + 1}")
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

    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    server = SimpleFederatedServer(
        bootstrap_servers=bootstrap_servers,
        model_topic='model_topic',
        update_topic='update_topic',
    )

    server.start()
