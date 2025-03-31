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
        # Perform federated averaging
        logging.info("Performing federated averaging")
        logging.info(f"Client updates received: {client_updates}")
        
        if client_updates is None or len(client_updates) == 0:
            logging.warning("No client updates received for federated averaging")
            return self.global_model
        client_updates = np.array(client_updates)
        if client_updates.ndim == 1:
            client_updates = client_updates.reshape(1, -1)
        if client_updates.shape[0] == 1:
            return client_updates[0]
        if client_updates.shape[1] != len(self.global_model):
            logging.error("Client updates shape mismatch with global model")
            return self.global_model
        # Perform averaging
        client_updates = np.array(client_updates)
        client_updates = np.mean(client_updates, axis=0)
        # Check if client_updates is empty
        if not client_updates:
            return self.global_model
        self.global_model = np.mean(client_updates, axis=0)
        logging.info(f"Global model updated with federated averaging: {self.global_model}")
        return self.global_model

    def send_model(self):
        self.producer.send(self.model_topic, np.array(self.global_model).astype(np.float32))
        self.producer.flush()
        logging.info("Global model sent to all clients")

    def start(self):
        if not self.connect_kafka():
            logging.error("Failed to start server due to Kafka connection issues")
            return

        try:
            for round in range(self.num_rounds):
                logging.info(f"Round {round + 1}/{self.num_rounds}")
            client_updates = []

            # Collect updates from clients - continuously listen for messages
            self.consumer.subscribe(['update_topic'])  # Ensure the consumer is subscribed
            for message in self.consumer:
                client_update = message.value
                client_id = self.assign_client_id()  # Assign client ID upon receiving update
                client_updates.append(client_update)
                logging.info(f"Server: Received update from client {client_id}: {client_update}")

                #  Consider adding a break condition if you want to limit updates per round
                #  For example, break after receiving a certain number of updates

            # Update global model
            self.global_model = self.federated_averaging(client_updates)
            self.send_model()
            logging.info(f"Round {round + 1} completed")

        except Exception as e:
            logging.error(f"An error occurred: {e}")
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