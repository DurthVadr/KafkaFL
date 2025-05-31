#!/usr/bin/env python3
"""
Simple Flower Server for CIFAR-10 with LeNet
"""

import flwr as fl
import tensorflow as tf
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model import create_lenet_model
from common.data import load_cifar10_data
from common.logger import get_server_logger

# Configure TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def get_evaluate_fn(model, test_data):
    """Return evaluation function for server-side evaluation"""
    X_test, y_test = test_data
    
    def evaluate(server_round, weights, config):
        """Evaluate global model on server's test set"""
        model.set_weights(weights)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Server eval round {server_round}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}
    
    return evaluate

def main():
    """Start simple Flower server"""
    print("Starting Simple Flower Server...")
    
    # Load test data
    print("Loading test data...")
    try:
        _, _, X_test, y_test = load_cifar10_data(subset_size=5000, test_size=1000)
        print(f"Test data loaded: {X_test.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create model
    print("Creating LeNet model...")
    try:
        model = create_lenet_model()
        print(f"Model created: {model.count_params()} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model, (X_test, y_test)),
    )
    
    # Server config
    config = fl.server.ServerConfig(num_rounds=25)
    
    print("Starting server on localhost:8080...")
    print("Waiting for at least 2 clients...")
    
    try:
        fl.server.start_server(
            server_address="localhost:8080",
            config=config,
            strategy=strategy,
        )
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()