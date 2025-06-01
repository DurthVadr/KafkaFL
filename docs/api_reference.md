# API Reference

This document provides a comprehensive API reference for the federated learning system, covering all modules, classes, functions, and their usage.

## Table of Contents

1. [Server API](#server-api)
2. [Client API](#client-api)
3. [Common Modules API](#common-modules-api)
4. [Kafka Integration API](#kafka-integration-api)
5. [Flower Framework API](#flower-framework-api)
6. [Monitoring API](#monitoring-api)
7. [Configuration API](#configuration-api)
8. [Utility Functions](#utility-functions)
9. [Error Handling](#error-handling)
10. [REST API Endpoints](#rest-api-endpoints)

## Server API

### FederatedServer Class

```python
class FederatedServer:
    """Main federated learning server implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize federated server.
        
        Args:
            config: Server configuration dictionary
                - model_config: Model architecture configuration
                - kafka_config: Kafka connection settings
                - aggregation_config: Aggregation parameters
                - logging_config: Logging settings
        """
        
    async def start(self) -> None:
        """
        Start the federated learning server.
        
        Raises:
            ConnectionError: If Kafka connection fails
            ConfigurationError: If configuration is invalid
        """
        
    async def stop(self) -> None:
        """Stop the server gracefully."""
        
    async def handle_client_update(self, client_id: str, update_data: Dict) -> bool:
        """
        Handle incoming client model update.
        
        Args:
            client_id: Unique client identifier
            update_data: Client update containing weights and metadata
                - weights: Model weights as numpy array or tensor
                - num_samples: Number of training samples
                - loss: Training loss value
                - accuracy: Training accuracy
                - metadata: Additional client information
        
        Returns:
            bool: True if update was processed successfully
            
        Raises:
            ValidationError: If update data is invalid
            AggregationError: If aggregation fails
        """
        
    def get_global_model(self) -> Dict[str, Any]:
        """
        Get current global model.
        
        Returns:
            Dict containing:
                - weights: Global model weights
                - version: Model version number
                - round: Current training round
                - metadata: Model metadata
        """
        
    def get_server_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Dict containing:
                - active_clients: Number of active clients
                - total_rounds: Total training rounds completed
                - current_accuracy: Current global model accuracy
                - aggregation_time: Last aggregation time
        """
```

### ModelAggregator Class

```python
class ModelAggregator:
    """Handles model aggregation strategies."""
    
    def __init__(self, strategy: str = "fedavg", **kwargs):
        """
        Initialize model aggregator.
        
        Args:
            strategy: Aggregation strategy ("fedavg", "fedprox", "scaffold")
            **kwargs: Strategy-specific parameters
        """
        
    def aggregate(self, client_updates: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate client model updates.
        
        Args:
            client_updates: List of client updates
                Each update contains:
                - client_id: Client identifier
                - weights: Model weights
                - num_samples: Number of training samples
                - metadata: Additional information
        
        Returns:
            Dict containing aggregated model:
                - weights: Aggregated model weights
                - participants: List of participating clients
                - aggregation_metadata: Aggregation information
                
        Raises:
            AggregationError: If aggregation fails
            InsufficientDataError: If not enough updates provided
        """
        
    def set_strategy(self, strategy: str, **kwargs) -> None:
        """
        Change aggregation strategy.
        
        Args:
            strategy: New aggregation strategy
            **kwargs: Strategy parameters
        """
```

### ClientManager Class

```python
class ClientManager:
    """Manages client registration and lifecycle."""
    
    def register_client(self, client_id: str, capabilities: Dict) -> bool:
        """
        Register a new client.
        
        Args:
            client_id: Unique client identifier
            capabilities: Client capabilities
                - compute_power: Relative compute power (0.0-1.0)
                - memory_gb: Available memory in GB
                - network_speed: Network speed category
                - gpu_available: Whether GPU is available
        
        Returns:
            bool: True if registration successful
            
        Raises:
            DuplicateClientError: If client already registered
        """
        
    def unregister_client(self, client_id: str) -> bool:
        """
        Unregister a client.
        
        Args:
            client_id: Client identifier to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        
    def get_active_clients(self) -> List[str]:
        """
        Get list of active client IDs.
        
        Returns:
            List of active client identifiers
        """
        
    def select_clients(self, num_clients: int, strategy: str = "random") -> List[str]:
        """
        Select clients for training round.
        
        Args:
            num_clients: Number of clients to select
            strategy: Selection strategy ("random", "resource_based", "staleness_aware")
            
        Returns:
            List of selected client IDs
            
        Raises:
            InsufficientClientsError: If not enough clients available
        """
```

## Client API

### FederatedClient Class

```python
class FederatedClient:
    """Federated learning client implementation."""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            config: Client configuration
                - server_address: Server connection details
                - model_config: Local model configuration
                - training_config: Training parameters
                - data_config: Local data configuration
        """
        
    async def start(self) -> None:
        """
        Start the federated client.
        
        Raises:
            ConnectionError: If server connection fails
            InitializationError: If client initialization fails
        """
        
    async def stop(self) -> None:
        """Stop the client gracefully."""
        
    async def train_model(self, global_weights: Dict) -> Dict[str, Any]:
        """
        Train local model with global weights.
        
        Args:
            global_weights: Global model weights to start training from
            
        Returns:
            Dict containing training results:
                - updated_weights: Locally trained weights
                - num_samples: Number of training samples
                - loss: Final training loss
                - accuracy: Training accuracy
                - training_time: Time spent training
                
        Raises:
            TrainingError: If training fails
            DataError: If local data is insufficient
        """
        
    def load_data(self) -> Tuple[Any, Any]:
        """
        Load local training and validation data.
        
        Returns:
            Tuple of (training_data, validation_data)
            
        Raises:
            DataLoadError: If data loading fails
        """
        
    def evaluate_model(self, weights: Dict, test_data: Any = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            weights: Model weights to evaluate
            test_data: Optional test data (uses validation data if None)
            
        Returns:
            Dict containing evaluation metrics:
                - loss: Evaluation loss
                - accuracy: Evaluation accuracy
                - other_metrics: Additional metrics based on task
        """
        
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get client information and capabilities.
        
        Returns:
            Dict containing:
                - client_id: Client identifier
                - data_size: Size of local dataset
                - compute_resources: Available compute resources
                - model_architecture: Local model details
        """
```

### LocalTrainer Class

```python
class LocalTrainer:
    """Handles local model training."""
    
    def __init__(self, model: Any, optimizer_config: Dict):
        """
        Initialize local trainer.
        
        Args:
            model: Local model instance
            optimizer_config: Optimizer configuration
                - name: Optimizer name ("sgd", "adam", "adamw")
                - learning_rate: Learning rate
                - weight_decay: Weight decay parameter
                - momentum: Momentum (for SGD)
        """
        
    def train_epoch(self, data_loader: Any, num_epochs: int = 1) -> Dict[str, float]:
        """
        Train model for specified epochs.
        
        Args:
            data_loader: Training data loader
            num_epochs: Number of training epochs
            
        Returns:
            Dict containing training metrics:
                - loss: Average loss
                - accuracy: Training accuracy
                - epoch_time: Time per epoch
        """
        
    def validate(self, data_loader: Any) -> Dict[str, float]:
        """
        Validate model performance.
        
        Args:
            data_loader: Validation data loader
            
        Returns:
            Dict containing validation metrics
        """
```

## Common Modules API

### Serialization Module

```python
def serialize_weights(weights: Dict) -> bytes:
    """
    Serialize model weights for transmission.
    
    Args:
        weights: Model weights dictionary
        
    Returns:
        Serialized weights as bytes
        
    Raises:
        SerializationError: If serialization fails
    """

def deserialize_weights(data: bytes) -> Dict:
    """
    Deserialize model weights from bytes.
    
    Args:
        data: Serialized weights data
        
    Returns:
        Deserialized weights dictionary
        
    Raises:
        DeserializationError: If deserialization fails
    """

def compress_weights(weights: bytes, compression: str = "gzip") -> bytes:
    """
    Compress serialized weights.
    
    Args:
        weights: Serialized weights
        compression: Compression algorithm ("gzip", "lz4", "zstd")
        
    Returns:
        Compressed weights
    """

def decompress_weights(compressed_data: bytes, compression: str = "gzip") -> bytes:
    """
    Decompress weights data.
    
    Args:
        compressed_data: Compressed weights
        compression: Compression algorithm used
        
    Returns:
        Decompressed weights
    """
```

### Communication Module

```python
class KafkaProducer:
    """Kafka message producer for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Kafka producer.
        
        Args:
            config: Kafka configuration
                - bootstrap_servers: Kafka brokers
                - acks: Acknowledgment setting
                - retries: Number of retries
                - compression_type: Message compression
        """
        
    async def send_message(self, topic: str, message: Dict, key: str = None) -> bool:
        """
        Send message to Kafka topic.
        
        Args:
            topic: Kafka topic name
            message: Message data
            key: Optional message key
            
        Returns:
            bool: True if message sent successfully
            
        Raises:
            KafkaError: If message sending fails
        """

class KafkaConsumer:
    """Kafka message consumer for federated learning."""
    
    def __init__(self, topics: List[str], config: Dict[str, Any]):
        """
        Initialize Kafka consumer.
        
        Args:
            topics: List of topics to subscribe to
            config: Kafka configuration
        """
        
    async def consume_messages(self) -> AsyncGenerator[Dict, None]:
        """
        Consume messages from subscribed topics.
        
        Yields:
            Dict: Message data with metadata
        """
```

### Model Utils Module

```python
def create_model(architecture: str, **params) -> Any:
    """
    Create model instance based on architecture.
    
    Args:
        architecture: Model architecture name
        **params: Architecture-specific parameters
        
    Returns:
        Model instance
        
    Raises:
        UnsupportedArchitectureError: If architecture not supported
    """

def get_model_weights(model: Any) -> Dict:
    """
    Extract weights from model.
    
    Args:
        model: Model instance
        
    Returns:
        Dict containing model weights
    """

def set_model_weights(model: Any, weights: Dict) -> None:
    """
    Set model weights.
    
    Args:
        model: Model instance
        weights: Weights to set
        
    Raises:
        WeightCompatibilityError: If weights incompatible with model
    """

def calculate_model_size(model: Any) -> int:
    """
    Calculate model size in bytes.
    
    Args:
        model: Model instance
        
    Returns:
        Model size in bytes
    """
```

## Kafka Integration API

### Topic Management

```python
class TopicManager:
    """Manages Kafka topics for federated learning."""
    
    def __init__(self, admin_config: Dict):
        """
        Initialize topic manager.
        
        Args:
            admin_config: Kafka admin configuration
        """
        
    async def create_topics(self, topic_configs: List[Dict]) -> bool:
        """
        Create Kafka topics.
        
        Args:
            topic_configs: List of topic configurations
                Each config contains:
                - name: Topic name
                - num_partitions: Number of partitions
                - replication_factor: Replication factor
                - config: Topic-specific settings
        
        Returns:
            bool: True if all topics created successfully
        """
        
    async def delete_topics(self, topic_names: List[str]) -> bool:
        """
        Delete Kafka topics.
        
        Args:
            topic_names: List of topic names to delete
            
        Returns:
            bool: True if deletion successful
        """
        
    async def list_topics(self) -> List[str]:
        """
        List all available topics.
        
        Returns:
            List of topic names
        """
```

### Message Patterns

```python
class MessageHandler:
    """Handles federated learning message patterns."""
    
    def create_client_update_message(self, client_id: str, update_data: Dict) -> Dict:
        """
        Create client update message.
        
        Args:
            client_id: Client identifier
            update_data: Update data
            
        Returns:
            Formatted message dict
        """
        
    def create_global_model_message(self, model_data: Dict, metadata: Dict) -> Dict:
        """
        Create global model broadcast message.
        
        Args:
            model_data: Global model data
            metadata: Model metadata
            
        Returns:
            Formatted message dict
        """
        
    def validate_message(self, message: Dict, message_type: str) -> bool:
        """
        Validate message format.
        
        Args:
            message: Message to validate
            message_type: Expected message type
            
        Returns:
            bool: True if message is valid
            
        Raises:
            MessageValidationError: If message is invalid
        """
```

## Flower Framework API

### FlowerServer Integration

```python
class FlowerFederatedServer:
    """Flower-compatible federated server."""
    
    def __init__(self, strategy: Any, config: Dict):
        """
        Initialize Flower server.
        
        Args:
            strategy: Flower aggregation strategy
            config: Server configuration
        """
        
    def start_flower_server(self, server_address: str, num_rounds: int) -> None:
        """
        Start Flower server.
        
        Args:
            server_address: Server address and port
            num_rounds: Number of training rounds
        """

class FlowerClient:
    """Flower-compatible client implementation."""
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters for Flower.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of model parameter arrays
        """
        
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model with given parameters.
        
        Args:
            parameters: Global model parameters
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model with given parameters.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
```

## Monitoring API

### Metrics Collection

```python
class MetricsCollector:
    """Collects and manages federated learning metrics."""
    
    def __init__(self, config: Dict):
        """
        Initialize metrics collector.
        
        Args:
            config: Metrics configuration
        """
        
    def record_training_metric(self, client_id: str, metric_name: str, value: float) -> None:
        """
        Record training metric.
        
        Args:
            client_id: Client identifier
            metric_name: Name of the metric
            value: Metric value
        """
        
    def record_aggregation_metric(self, round_num: int, metric_data: Dict) -> None:
        """
        Record aggregation metrics.
        
        Args:
            round_num: Training round number
            metric_data: Aggregation metrics
        """
        
    def get_metrics_summary(self, time_range: Tuple[datetime, datetime] = None) -> Dict:
        """
        Get metrics summary.
        
        Args:
            time_range: Optional time range filter
            
        Returns:
            Dict containing metrics summary
        """
        
    def export_metrics(self, format: str = "prometheus") -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ("prometheus", "json", "csv")
            
        Returns:
            Formatted metrics string
        """
```

### Logging System

```python
class FederatedLogger:
    """Centralized logging for federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize logger.
        
        Args:
            config: Logging configuration
                - level: Log level
                - format: Log format
                - handlers: List of log handlers
        """
        
    def log_client_event(self, client_id: str, event: str, data: Dict = None) -> None:
        """
        Log client event.
        
        Args:
            client_id: Client identifier
            event: Event name
            data: Optional event data
        """
        
    def log_server_event(self, event: str, data: Dict = None) -> None:
        """
        Log server event.
        
        Args:
            event: Event name
            data: Optional event data
        """
        
    def log_aggregation_event(self, round_num: int, participants: List[str], metrics: Dict) -> None:
        """
        Log aggregation event.
        
        Args:
            round_num: Training round number
            participants: List of participating clients
            metrics: Aggregation metrics
        """
```

## Configuration API

### Configuration Manager

```python
class ConfigManager:
    """Manages federated learning configuration."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        
    def validate_config(self, config: Dict) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        
    def get_default_config(self, component: str) -> Dict:
        """
        Get default configuration for component.
        
        Args:
            component: Component name ("server", "client", "kafka", etc.)
            
        Returns:
            Default configuration dictionary
        """
        
    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
```

## Utility Functions

### Data Utilities

```python
def split_data(data: Any, num_clients: int, strategy: str = "iid") -> List[Any]:
    """
    Split data among clients.
    
    Args:
        data: Dataset to split
        num_clients: Number of clients
        strategy: Splitting strategy ("iid", "non_iid", "pathological")
        
    Returns:
        List of client datasets
    """

def create_non_iid_split(data: Any, num_clients: int, alpha: float = 0.5) -> List[Any]:
    """
    Create non-IID data split using Dirichlet distribution.
    
    Args:
        data: Dataset to split
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of non-IID client datasets
    """

def calculate_data_distribution(client_datasets: List[Any]) -> Dict[str, Any]:
    """
    Calculate data distribution statistics.
    
    Args:
        client_datasets: List of client datasets
        
    Returns:
        Distribution statistics
    """
```

### Security Utilities

```python
def generate_client_certificate(client_id: str) -> Tuple[str, str]:
    """
    Generate client certificate for secure communication.
    
    Args:
        client_id: Client identifier
        
    Returns:
        Tuple of (certificate, private_key)
    """

def validate_client_certificate(certificate: str, client_id: str) -> bool:
    """
    Validate client certificate.
    
    Args:
        certificate: Client certificate
        client_id: Expected client identifier
        
    Returns:
        bool: True if certificate is valid
    """

def encrypt_message(message: Dict, public_key: str) -> bytes:
    """
    Encrypt message for secure transmission.
    
    Args:
        message: Message to encrypt
        public_key: Recipient's public key
        
    Returns:
        Encrypted message bytes
    """

def decrypt_message(encrypted_data: bytes, private_key: str) -> Dict:
    """
    Decrypt received message.
    
    Args:
        encrypted_data: Encrypted message
        private_key: Recipient's private key
        
    Returns:
        Decrypted message
    """
```

## Error Handling

### Exception Classes

```python
class FederatedLearningError(Exception):
    """Base exception for federated learning errors."""
    pass

class ConfigurationError(FederatedLearningError):
    """Configuration-related errors."""
    pass

class ConnectionError(FederatedLearningError):
    """Network connection errors."""
    pass

class AggregationError(FederatedLearningError):
    """Model aggregation errors."""
    pass

class TrainingError(FederatedLearningError):
    """Local training errors."""
    pass

class SerializationError(FederatedLearningError):
    """Data serialization errors."""
    pass

class ValidationError(FederatedLearningError):
    """Data validation errors."""
    pass

class InsufficientDataError(FederatedLearningError):
    """Insufficient data for training."""
    pass

class ClientNotFoundError(FederatedLearningError):
    """Client not found in registry."""
    pass

class ModelCompatibilityError(FederatedLearningError):
    """Model compatibility issues."""
    pass
```

### Error Handler

```python
class ErrorHandler:
    """Centralized error handling for federated learning."""
    
    def __init__(self, logger: Any):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance
        """
        
    def handle_error(self, error: Exception, context: Dict = None) -> bool:
        """
        Handle and log error.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            bool: True if error was handled gracefully
        """
        
    def is_recoverable_error(self, error: Exception) -> bool:
        """
        Check if error is recoverable.
        
        Args:
            error: Exception to check
            
        Returns:
            bool: True if error is recoverable
        """
```

## REST API Endpoints

### Server Endpoints

```python
# GET /api/v1/server/status
{
    "status": "running",
    "active_clients": 15,
    "current_round": 42,
    "global_accuracy": 0.873,
    "uptime": "2h 15m 30s"
}

# GET /api/v1/server/model
{
    "model_version": 42,
    "architecture": "cnn",
    "accuracy": 0.873,
    "size_bytes": 1048576,
    "last_updated": "2025-01-01T12:00:00Z"
}

# POST /api/v1/server/config
{
    "aggregation_strategy": "fedavg",
    "min_clients": 10,
    "max_rounds": 100
}

# GET /api/v1/clients
[
    {
        "client_id": "client_001",
        "status": "active",
        "last_seen": "2025-01-01T12:00:00Z",
        "data_samples": 1000,
        "model_version": 42
    }
]

# GET /api/v1/metrics
{
    "training_metrics": {
        "global_loss": 0.234,
        "global_accuracy": 0.873,
        "convergence_rate": 0.95
    },
    "system_metrics": {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "network_throughput": 128.5
    }
}
```

### Client Endpoints

```python
# POST /api/v1/client/register
{
    "client_id": "client_001",
    "capabilities": {
        "compute_power": 0.8,
        "memory_gb": 8,
        "gpu_available": true
    }
}

# GET /api/v1/client/{client_id}/status
{
    "client_id": "client_001",
    "status": "training",
    "current_round": 42,
    "progress": 0.75,
    "local_accuracy": 0.891
}

# POST /api/v1/client/{client_id}/update
{
    "model_version": 42,
    "weights": "base64_encoded_weights",
    "num_samples": 1000,
    "training_metrics": {
        "loss": 0.234,
        "accuracy": 0.891,
        "training_time": 45.2
    }
}
```

This comprehensive API reference provides detailed documentation for all components of the federated learning system, enabling developers to integrate, extend, and customize the system according to their needs.
