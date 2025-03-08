
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
            value_deserializer=lambda m: np.array(m)
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.tobytes()
        )
        self.update_topic = update_topic
        self.client_id = 'client_1'
        
        self.X, self.y = self.load_data()
        

    
    def load_data_cifar10(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        return X_train, y_train, X_test, y_test
        
    
    def train(self):
        logging.info("Training model")
        logging.info("model weights before training: {}".format(self.model))
        X_train, y_train, X_test, y_test = self.load_data_cifar10()
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: {}".format(accuracy))
        self.model = model.coef_
        logging.info("model weights after training: {}".format(self.model))
        return self.model
    
        
       
            
    
    def send_update(self):
        self.producer.send(self.update_topic, self.model)
        
    def consume_model_from_topic(self):
        for message in self.consumer:
            self.model = message.value
            break
        logging.info("Model received from server")
        logging.info("model weights: {}".format(self.model))
        
    def start(self):
        self.consume_model_from_topic()
        self.train()
        self.send_update()
        logging.info("Model update sent to server")
        
        
        
if __name__ == "__main__":
    
    print("Client started")
    
    client = FederatedClient(
        bootstrap_servers='localhost:9092',
        update_topic='update_topic',
        model_topic='model_topic'
    )
    client.start()
   