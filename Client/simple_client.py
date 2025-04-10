from kafka import KafkaConsumer, KafkaProducer
import numpy as np
import logging
import time
import os
import traceback

class SimpleFederatedClient:
    _client_id_counter = 0  # Class-level counter for unique client IDs

    def __init__(self, bootstrap_servers, update_topic, model_topic, client_id=None):
        self.bootstrap_servers = bootstrap_servers
        self.update_topic = update_topic
        self.model_topic = model_topic

        # Use provided client_id if available, otherwise generate one
        if client_id is not None:
            self.client_id = int(client_id)
        else:
            self.client_id = SimpleFederatedClient.generate_client_id()  # Assign unique ID

        self.model = None
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
                # Create consumer with better configuration
                self.consumer = KafkaConsumer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32),
                    auto_offset_reset='latest',  # Start from the latest message
                    group_id=f'client-{self.client_id}',
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000,
                    max_poll_interval_ms=300000,  # 5 minutes
                    fetch_max_wait_ms=500,
                    fetch_min_bytes=1,
                    fetch_max_bytes=52428800  # 50MB
                )

                # Create producer with better configuration
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: v.tobytes(),
                    acks='all',  # Wait for all replicas
                    retries=5,  # Retry 5 times
                    batch_size=16384,  # 16KB batches
                    linger_ms=100,  # Wait 100ms to batch messages
                    buffer_memory=33554432  # 32MB buffer
                )

                # Subscribe to the model topic
                self.consumer.subscribe([self.model_topic])
                logging.info(f"Client {self.client_id}: Subscribed to topic {self.model_topic}")

                # Test connection by polling
                poll_result = self.consumer.poll(timeout_ms=100)
                logging.info(f"Client {self.client_id}: Initial poll result: {poll_result}")

                logging.info(f"Client {self.client_id}: Successfully connected to Kafka at {self.bootstrap_servers}")
                return True
            except Exception as e:
                logging.error(f"Client {self.client_id}: Failed to connect to Kafka (attempt {attempt+1}/{max_attempts}): {e}")
                logging.error(traceback.format_exc())
                attempt += 1
                if attempt < max_attempts:
                    logging.info(f"Client {self.client_id}: Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        return False

    def train(self, global_weights):
        logging.info(f"Client {self.client_id}: Training model (simplified)")

        # Simulate training by adding random noise to the global model
        # This is a simplified version that doesn't require TensorFlow
        local_update = global_weights + np.random.normal(0, 0.1, size=global_weights.shape)
        self.model = local_update

        # Simulate accuracy
        accuracy = np.random.uniform(0.7, 0.95)
        logging.info(f"Client {self.client_id}: Model accuracy after training: {accuracy:.4f}")

        return self.model

    def send_update(self):
        try:
            # Send the model update
            future = self.producer.send(self.update_topic, self.model)
            record_metadata = future.get(timeout=10)
            self.producer.flush()
            logging.info(f"Client {self.client_id}: Model update sent to server. Size: {self.model.size}")
            logging.info(f"Client {self.client_id}: Message sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")
            return True
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending update: {e}")
            logging.error(traceback.format_exc())
            return False

    def consume_model_from_topic(self):
        try:
            # We already subscribed in connect_kafka, but let's make sure
            self.consumer.subscribe([self.model_topic])
            logging.info(f"Client {self.client_id}: Ensuring subscription to topic {self.model_topic}")

            # Set a timeout for receiving the model (60 seconds)
            timeout_ms = 60000
            start_time = time.time()
            poll_interval = 5000  # 5 seconds

            # First, check if there are any existing messages
            logging.info(f"Client {self.client_id}: Checking for existing messages in {self.model_topic}")
            poll_result = self.consumer.poll(timeout_ms=1000)
            if poll_result:
                logging.info(f"Client {self.client_id}: Found existing messages: {poll_result}")
            else:
                logging.info(f"Client {self.client_id}: No existing messages found, waiting for new ones")

            # Main polling loop
            poll_count = 0
            while (time.time() - start_time) < (timeout_ms/1000):
                # Poll for messages with a timeout
                poll_count += 1
                logging.info(f"Client {self.client_id}: Polling for messages (attempt {poll_count})")
                poll_result = self.consumer.poll(timeout_ms=poll_interval)

                if poll_result:
                    logging.info(f"Client {self.client_id}: Received poll result with {sum(len(msgs) for msgs in poll_result.values())} messages")
                    for tp, messages in poll_result.items():
                        logging.info(f"Client {self.client_id}: Processing messages from topic-partition: {tp.topic}-{tp.partition}")
                        for message in messages:
                            try:
                                # Get the model weights
                                model_weights = message.value
                                self.model = model_weights
                                logging.info(f"Client {self.client_id}: Model received from server. Size: {model_weights.size}")
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

    def start(self, num_rounds=10):
        logging.info(f"Client {self.client_id}: Starting")
        if not self.connect_kafka():
            logging.error(f"Client {self.client_id}: Failed to connect to Kafka, exiting")
            return

        for round in range(num_rounds):
            logging.info(f"Client {self.client_id}: Starting round {round + 1}")

            # Wait for the global model
            global_model = self.consume_model_from_topic()
            if global_model is None:
                logging.error(f"Client {self.client_id}: Failed to receive model, exiting round")
                continue

            # Train the local model
            local_update = self.train(global_model)
            if local_update is None:
                logging.error(f"Client {self.client_id}: Training failed, exiting round")
                continue

            # Send the update to the server
            if self.send_update():
                logging.info(f"Client {self.client_id}: Round {round + 1} completed successfully")
            else:
                logging.error(f"Client {self.client_id}: Failed to send update in round {round + 1}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    client_id = os.environ.get("CLIENT_ID", None)

    if client_id:
        logging.info(f"Starting client with ID: {client_id}")
    else:
        logging.info("No client ID provided, will generate one automatically")

    client = SimpleFederatedClient(
        bootstrap_servers=bootstrap_servers,
        update_topic='update_topic',
        model_topic='model_topic',
        client_id=client_id
    )

    client.start(num_rounds=10)
