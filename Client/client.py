from kafka import KafkaConsumer, KafkaProducer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
import numpy as np
import logging
import time
import os

class FederatedClient:
    _client_id_counter = 0  # Class-level counter for unique client IDs

    def __init__(self, bootstrap_servers, update_topic, model_topic):
        self.bootstrap_servers = bootstrap_servers
        self.update_topic = update_topic
        self.model_topic = model_topic
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

    def train(self, global_model):
        logging.info(f"Client {self.client_id}: Training model")
        #model = LogisticRegression(max_iter=1000)
        #model.fit(self.X, self.y)
        #y_pred = model.predict(self.X)
        #accuracy = accuracy_score(self.y, y_pred)
        #logging.info(f"Client {self.client_id}: Model accuracy: {accuracy}")
        #self.model = model.coef_.flatten()  # Flatten the model weights
        
        # Placeholder for training with the global model
        # Replace this with your actual training logic
        local_update = global_model + np.random.normal(0, 0.1, size=global_model.shape)  # Simulate an update
        self.model = local_update
        return self.model

    def send_update(self):
        try:
            self.producer.send(self.update_topic, self.model)
            self.producer.flush()
            logging.info(f"Client {self.client_id}: Model update sent to server")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error sending update: {e}")

    def consume_model_from_topic(self):
        try:
            self.consumer.subscribe([self.model_topic])
            logging.info(f"Client {self.client_id}: Subscribed to topic {self.model_topic}")
            for message in self.consumer:
                self.model = message.value
                logging.info(f"Client {self.client_id}: Model received from server")
                logging.info(f"Client {self.client_id}: Model weights: {self.model}")
                return self.model  # Return the model after receiving it
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error consuming model: {e}")
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
    
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9092")
    client = FederatedClient(
        bootstrap_servers=bootstrap_servers,
        update_topic='update_topic',
        model_topic='model_topic',
        
    )
    client.start(num_rounds=10)