from kafka import KafkaConsumer, KafkaProducer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
import numpy as np
import logging

class FederatedClient:
    def __init__(self, bootstrap_servers, update_topic, model_topic):
        self.consumer = KafkaConsumer(
            model_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32)
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.tobytes()
        )
        self.update_topic = update_topic
        self.client_id = None
        self.model = None
        self.X, self.y = self.load_data_cifar10()

    def load_data_cifar10(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        return X_train, y_train

    def train(self):
        logging.info(f"Client {self.client_id}: Training model")
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        logging.info(f"Client {self.client_id}: Model accuracy: {accuracy}")
        self.model = model.coef_.flatten()  # Flatten the model weights
        return self.model

    def send_update(self):
        self.producer.send(self.update_topic, self.model)
        self.producer.flush()
        logging.info(f"Client {self.client_id}: Model update sent to server")

    def consume_model_from_topic(self):
        for message in self.consumer:
            self.model = message.value
            logging.info(f"Client {self.client_id}: Model received from server")
            logging.info(f"Client {self.client_id}: Model weights: {self.model}")
            break


    def register_client(self):
        # Send a registration request to the server
        self.producer.send(self.update_topic, np.array([0]))  # Sending a dummy message for registration
        self.producer.flush()
        logging.info("Client: Registration request sent to server")
        
        
        
    def start(self):
        self.register_client()
        self.consume_model_from_topic()
        self.train()
        self.send_update()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    
    
    client = FederatedClient(
        bootstrap_servers='localhost:9092',
        update_topic='update_topic',
        model_topic='model_topic',
        client_id=client_id
    )
    client.start()