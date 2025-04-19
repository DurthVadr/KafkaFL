"""
Federated Learning Server

This module implements a server for federated learning using Kafka for communication.
The server aggregates model updates from clients and distributes the global model.
"""

import os
import time
import signal
import sys
import numpy as np

# Import common modules
from common.logger import get_server_logger
from common.model import create_cifar10_model, get_random_weights
from common.serialization import serialize_weights, deserialize_weights
from common.kafka_utils import create_producer, create_consumer, send_message, receive_messages, close_kafka_resources

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class FederatedServer:
    """Server for federated learning"""
    
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        """
        Initialize the federated learning server.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            model_topic: Topic for sending global model to clients
            update_topic: Topic for receiving model updates from clients
        """
        # Initialize logger
        self.logger = get_server_logger()
        self.logger.info("Initializing federated learning server")
        
        # Set Kafka configuration
        self.bootstrap_servers = bootstrap_servers
        self.model_topic = model_topic
        self.update_topic = update_topic
        self.producer = None
        self.consumer = None
        
        # Initialize global model
        self.global_model = self._initialize_global_model()
        
        # Connect to Kafka
        self._connect_to_kafka()
    
    def _initialize_global_model(self):
        """
        Initialize the global model.
        
        Returns:
            List of weight arrays representing the global model
        """
        self.logger.info("Initializing global model")
        
        if TENSORFLOW_AVAILABLE:
            try:
                # Create model
                model = create_cifar10_model()
                
                if model is not None:
                    # Get initial weights
                    weights = model.get_weights()
                    self.logger.info(f"Initialized global model with {len(weights)} layers")
                    return weights
            except Exception as e:
                self.logger.error(f"Error initializing global model: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        
        # Fallback to random weights
        self.logger.warning("Using random weights for global model")
        weights = get_random_weights()
        self.logger.info(f"Initialized random global model with {len(weights)} layers")
        return weights
    
    def _connect_to_kafka(self):
        """Connect to Kafka broker"""
        self.logger.info(f"Connecting to Kafka at {self.bootstrap_servers}")
        
        # Create producer
        self.producer = create_producer(
            bootstrap_servers=self.bootstrap_servers,
            logger=self.logger
        )
        
        # Create consumer
        self.consumer = create_consumer(
            bootstrap_servers=self.bootstrap_servers,
            group_id="federated_server",
            topics=[self.update_topic],
            logger=self.logger
        )
        
        if self.producer is not None and self.consumer is not None:
            self.logger.info("Successfully connected to Kafka")
            return True
        else:
            self.logger.error("Failed to connect to Kafka")
            return False
    
    def send_global_model(self):
        """
        Send the global model to clients.
        
        Returns:
            True if the model was sent successfully, False otherwise
        """
        self.logger.info(f"Sending global model to clients on topic {self.model_topic}")
        
        # Serialize the global model
        serialized_model = serialize_weights(self.global_model, logger=self.logger)
        
        if serialized_model is None:
            self.logger.error("Failed to serialize global model")
            return False
        
        # Send the global model
        success = send_message(
            producer=self.producer,
            topic=self.model_topic,
            message=serialized_model,
            logger=self.logger
        )
        
        if success:
            self.logger.info("Global model sent successfully")
        else:
            self.logger.error("Failed to send global model")
        
        return success
    
    def receive_model_updates(self, max_updates=3, timeout_ms=60000):
        """
        Receive model updates from clients.
        
        Args:
            max_updates: Maximum number of updates to receive (default: 3)
            timeout_ms: Timeout in milliseconds (default: 60000)
            
        Returns:
            List of model updates, or empty list if no updates were received
        """
        self.logger.info(f"Waiting for up to {max_updates} model updates from clients on topic {self.update_topic}")
        
        # Receive messages from the update topic
        messages = receive_messages(
            consumer=self.consumer,
            timeout_ms=timeout_ms,
            max_messages=max_updates,
            logger=self.logger
        )
        
        if not messages:
            self.logger.warning("No model updates received from clients")
            return []
        
        # Deserialize model updates
        updates = []
        for i, message in enumerate(messages):
            update = deserialize_weights(message, logger=self.logger)
            
            if update is not None:
                self.logger.info(f"Deserialized update {i+1}/{len(messages)} with {len(update)} layers")
                updates.append(update)
            else:
                self.logger.error(f"Failed to deserialize update {i+1}/{len(messages)}")
        
        self.logger.info(f"Received {len(updates)} valid model updates from clients")
        return updates
    
    def aggregate_model_updates(self, updates):
        """
        Aggregate model updates using federated averaging.
        
        Args:
            updates: List of model updates
            
        Returns:
            Aggregated model weights
        """
        if not updates:
            self.logger.warning("No updates to aggregate. Keeping current global model.")
            return self.global_model
        
        self.logger.info(f"Aggregating {len(updates)} model updates")
        
        try:
            # Get the number of layers in the model
            num_layers = len(self.global_model)
            
            # Check if all updates have the same number of layers
            if not all(len(update) == num_layers for update in updates):
                self.logger.error("Updates have different numbers of layers. Cannot aggregate.")
                return self.global_model
            
            # Initialize aggregated weights
            aggregated_weights = []
            
            # Aggregate each layer
            for layer_idx in range(num_layers):
                # Get weights for this layer from all updates
                layer_weights = [update[layer_idx] for update in updates]
                
                # Check if all weights have the same shape
                if not all(w.shape == layer_weights[0].shape for w in layer_weights):
                    self.logger.error(f"Layer {layer_idx} has inconsistent shapes across updates. Cannot aggregate.")
                    return self.global_model
                
                # Average the weights
                layer_avg = np.mean(layer_weights, axis=0)
                aggregated_weights.append(layer_avg)
            
            self.logger.info(f"Successfully aggregated model updates into a new global model with {len(aggregated_weights)} layers")
            return aggregated_weights
        except Exception as e:
            self.logger.error(f"Error aggregating model updates: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self.global_model
    
    def start(self, num_rounds=10):
        """
        Start the federated learning process.
        
        Args:
            num_rounds: Number of federated learning rounds (default: 10)
        """
        self.logger.info(f"Starting federated learning server with {num_rounds} rounds")
        
        # Send initial global model
        if not self.send_global_model():
            self.logger.error("Failed to send initial global model. Exiting.")
            return
        
        # Run federated learning rounds
        for round_idx in range(num_rounds):
            self.logger.info(f"=== Round {round_idx + 1}/{num_rounds} ===")
            
            # Receive model updates from clients
            updates = self.receive_model_updates(max_updates=3, timeout_ms=120000)
            
            if not updates:
                self.logger.warning(f"No updates received in round {round_idx + 1}. Skipping aggregation.")
                continue
            
            # Aggregate model updates
            self.global_model = self.aggregate_model_updates(updates)
            
            # Send updated global model to clients
            if not self.send_global_model():
                self.logger.error(f"Failed to send global model in round {round_idx + 1}")
                continue
            
            self.logger.info(f"Completed round {round_idx + 1}/{num_rounds}")
        
        self.logger.info("Federated learning completed")
    
    def close(self):
        """Close resources"""
        self.logger.info("Closing server resources")
        
        # Close Kafka resources
        close_kafka_resources(
            producer=self.producer,
            consumer=self.consumer,
            logger=self.logger
        )
        
        # Clear model
        self.global_model = None
        
        self.logger.info("Server resources closed")

# Global variable for signal handling
server_instance = None

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("Received termination signal. Closing server gracefully...")
    if server_instance is not None:
        server_instance.close()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration from environment variables
    bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
    
    # Create and start server
    server = FederatedServer(
        bootstrap_servers=bootstrap_servers,
        model_topic="global_model",
        update_topic="model_updates"
    )
    
    # Store server instance for signal handling
    server_instance = server
    
    # Start federated learning
    server.start(num_rounds=10)
    
    # Close resources
    server.close()
