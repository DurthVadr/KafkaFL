
from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np
import logging
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model




class FederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic,):
        self.consumer = KafkaConsumer(
            update_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.model_topic = model_topic
        self.global_model = self.initialize_global_model_cifar10()
        self.client_updates = []
        self.clients = []
        self.num_rounds = 10
        self.round = 0
        self.strategy = 'federated_averaging'
        
        #TODO Implement other strategies and asynchrounous federated learning
        
        if self.strategy == 'federated_averaging':
            self.federated_strategy = self.federated_averaging
        elif self.strategy == 'federated_median':
            self.federated_strategy = self.federated_median
    
        
        
        
        
    
    
    def consume_model(self):
        for message in self.consumer:
            self.global_model = message.value
            break
        logging.info("Model received from client")
        
        
        logging.info("model weights: {}".format(self.global_model))
        
        
    
    def federated_averaging(self):
        # Perform federated averaging
        pass
    
    def federated_median(self):
        # Perform federated median
        pass
    
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
        
        logging.info("Global Model cifar10 initialized. Weights: {}".format(model.get_weights()))
        return model
        
    def update_model(self, client_update):
        # Update the global model with the client update
        
        logging.info("Updating model")
        logging.info("client update: {}".format(client_update))
        logging.info("global model: {}".format(self.global_model))
        
        pass

    def send_model(self):
        # Send the global model weights to all clients
        logging.info("Sending model")
        self.producer.send(self.model_topic, self.global_model)
        
        self.producer.flush()
        logging.info("Model sent to all clients")
        logging.info("model weights: {}".format(self.global_model))
        
    
        
        
    def start(self):
        for i in range(self.num_rounds):
            logging.info("Round: {}".format(i))
            self.round = i
            self.consume_model()
            self.federated_averaging()
            self.send_model()
            logging.info("Model update sent to all clients")
            logging.info("Waiting for updates")
            logging.info("Round completed")
            logging.info("====================================")
            
        
        
      
if __name__ == "__main__":
    
    
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Server started")
    logging.info("Listening for updates")
    
    server = FederatedServer(
        bootstrap_servers='localhost:9092',
        model_topic='model_topic',
        update_topic='update_topic',
        
        
        
    )
    
    server.start()
    
    
"""

from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Sending model updates to Kafka topic
model_update = {'model_id': '123', 'weights': [0.1, 0.2, 0.3]}
producer.send('model_updates', model_update)
producer.flush()

"""