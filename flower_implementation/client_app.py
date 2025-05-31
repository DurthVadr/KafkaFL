#!/usr/bin/env python3
"""
Simple Flower Client for CIFAR-10 with LeNet
"""

import flwr as fl
import tensorflow as tf
import numpy as np
import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model import create_lenet_model
from common.data import load_cifar10_data

# Configure TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class SimpleFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy ndarrays"""
        return self.model.get_weights()

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays"""
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Train the model on the locally held training set"""
        print(f"Client {self.client_id}: Starting training...")
        self.set_parameters(parameters)
        
        # Train model
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        
        print(f"Client {self.client_id}: Training completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self.get_parameters(config), len(self.X_train), {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        """Evaluate the model on the locally held test set"""
        self.set_parameters(parameters)
        
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"Client {self.client_id}: Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.X_test), {"accuracy": accuracy}

def partition_data(X_train, y_train, X_test, y_test, client_id, num_clients):
    """Partition data for federated learning"""
    # Simple IID partitioning
    train_size = len(X_train) // num_clients
    test_size = len(X_test) // num_clients
    
    train_start = client_id * train_size
    train_end = train_start + train_size
    
    test_start = client_id * test_size
    test_end = test_start + test_size
    
    return (X_train[train_start:train_end], y_train[train_start:train_end],
            X_test[test_start:test_end], y_test[test_start:test_end])

def main():
    """Start simple Flower client"""
    parser = argparse.ArgumentParser(description="Simple Flower Client")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID")
    parser.add_argument("--num_clients", type=int, default=3, help="Total number of clients")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    args = parser.parse_args()
    
    print(f"Starting Simple Flower Client {args.client_id}...")
    
    # Load data
    print("Loading CIFAR-10 data...")
    try:
        X_train, y_train, X_test, y_test = load_cifar10_data(subset_size=15000, test_size=3000)
        print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Partition data
    print(f"Partitioning data for client {args.client_id}...")
    X_train_client, y_train_client, X_test_client, y_test_client = partition_data(
        X_train, y_train, X_test, y_test, args.client_id, args.num_clients
    )
    
    print(f"Client {args.client_id} data - Train: {X_train_client.shape}, Test: {X_test_client.shape}")
    
    # Create model
    print("Creating LeNet model...")
    try:
        model = create_lenet_model()
        print(f"Model created: {model.count_params()} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Create client
    client = SimpleFlowerClient(
        model, X_train_client, y_train_client, X_test_client, y_test_client, args.client_id
    )
    
    print(f"Connecting to server at {args.server}...")
    
    try:
        fl.client.start_numpy_client(
            server_address=args.server,
            client=client
        )
    except Exception as e:
        print(f"Client error: {e}")

if __name__ == "__main__":
    main()