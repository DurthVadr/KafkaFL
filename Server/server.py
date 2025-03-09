from kafka import KafkaConsumer, KafkaProducer
import numpy as np
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class FederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        self.consumer = KafkaConsumer(
            update_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: np.frombuffer(m, dtype=np.float32)
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: v.tobytes()
        )
        self.model_topic = model_topic
        self.global_model = self.initialize_global_model_cifar10()
        self.num_rounds = 10
        self.client_id_counter = 0  # Counter for client IDs

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
        for round in range(self.num_rounds):
            logging.info(f"Round {round + 1}/{self.num_rounds}")
            client_updates = []

            # Collect updates from clients
            for message in self.consumer:
                client_update = message.value
                client_id = self.assign_client_id()  # Assign client ID upon receiving update
                client_updates.append(client_update)
                logging.info(f"Server: Received update from client {client_id}: {client_update}")

            # Update global model
            self.global_model = self.federated_averaging(client_updates)
            self.send_model()
            logging.info(f"Round {round + 1} completed")

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Server started")
    
    server = FederatedServer(
        bootstrap_servers='localhost:9092',
        model_topic='model_topic',
        update_topic='update_topic',
    )
    
    server.start()