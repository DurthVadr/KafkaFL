# System Architecture

This document provides a comprehensive overview of the federated learning system architecture, including component interactions, data flow, and design decisions.

## Architecture Overview

The system follows a distributed architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Federated Learning System                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐       │
│  │   Client 1  │    │    Apache    │    │   Server    │       │
│  │             │◄──►│    Kafka     │◄──►│             │       │
│  └─────────────┘    │   (Broker)   │    └─────────────┘       │
│                     │              │                          │
│  ┌─────────────┐    │  Topics:     │    ┌─────────────┐       │
│  │   Client 2  │◄──►│ - global_model   │ │ Visualization│       │
│  │             │    │ - model_updates  │ │   Module    │       │
│  └─────────────┘    │              │    └─────────────┘       │
│                     │              │                          │
│  ┌─────────────┐    └──────────────┘    ┌─────────────┐       │
│  │   Client N  │                        │  Monitoring │       │
│  │             │                        │   Module    │       │
│  └─────────────┘                        └─────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Federated Server (`server.py`)

The server is the central coordinator of the federated learning process.

#### Core Responsibilities
- **Global Model Management**: Maintains and updates the global model
- **Client Coordination**: Orchestrates the federated learning rounds
- **Model Aggregation**: Combines client updates using federated averaging
- **Communication**: Manages Kafka-based messaging with clients

#### Class Structure
```python
class FederatedServer:
    def __init__(self, bootstrap_servers, model_topic, update_topic)
    def _initialize_global_model(self)
    def _connect_to_kafka(self)
    def send_global_model(self)
    def receive_model_updates(self, max_updates, timeout_ms)
    def aggregate_model_updates(self, updates)
    def start(self, duration_minutes, aggregation_interval_seconds, min_updates_per_aggregation)
    def generate_visualizations(self)
    def close(self)
```

#### Key Features
- **Asynchronous Operation**: Time-based aggregation intervals
- **Flexible Aggregation**: Configurable minimum updates per round
- **Metrics Tracking**: Comprehensive performance monitoring
- **Fault Tolerance**: Robust error handling and recovery

### 2. Federated Client (`client.py`)

Clients perform local training and communicate with the server.

#### Core Responsibilities
- **Local Training**: Train models on local CIFAR-10 data subsets
- **Model Synchronization**: Receive global models and send updates
- **Weight Adaptation**: Handle model architecture compatibility
- **Performance Monitoring**: Track training metrics and accuracy

#### Class Structure
```python
class FederatedClient:
    def __init__(self, bootstrap_servers, update_topic, model_topic, client_id)
    def _connect_to_kafka(self)
    def receive_global_model(self)
    def train_local_model(self, global_weights)
    def send_model_update(self, weights)
    def start(self, duration_minutes, training_interval_seconds)
    def generate_visualizations(self)
    def close(self)
```

#### Key Features
- **Independent Operation**: Autonomous training cycles
- **Data Privacy**: Local data never leaves the client
- **Adaptive Training**: Dynamic interval adjustment
- **Comprehensive Metrics**: Detailed performance tracking

### 3. Common Modules (`common/`)

Shared functionality across the system.

#### Data Module (`common/data.py`)
```python
def load_cifar10_data(subset_size=None, test_size=None, logger=None)
def preprocess_data(X, y)
def create_data_partitions(X, y, num_clients)
```

#### Model Module (`common/model.py`)
```python
def create_lenet_model()
def create_cnn_model()
def are_weights_compatible(model, weights)
def adapt_weights(model, weights)
def get_random_weights(model_type)
```

#### Kafka Utils (`common/kafka_utils.py`)
```python
def create_producer(bootstrap_servers, logger=None)
def create_consumer(bootstrap_servers, group_id, topics, logger=None)
def send_message(producer, topic, message, logger=None)
def receive_messages(consumer, timeout_ms, max_messages, logger=None)
def close_kafka_resources(producer, consumer, logger=None)
```

#### Serialization Module (`common/serialization.py`)
```python
def serialize_weights(weights, logger=None)
def deserialize_weights(data, logger=None)
def compress_data(data)
def decompress_data(data)
```

#### Visualization Module (`common/visualization.py`)
```python
def plot_client_accuracy(accuracy_data, client_id, logger=None)
def plot_client_loss(loss_data, client_id, logger=None)
def plot_server_aggregations(times, counts, logger=None)
def plot_weight_distribution_violin(weights, round_num, logger=None)
def plot_convergence_visualization(weight_history, layer_idx, logger=None)
def plot_client_similarity_heatmap(client_updates, logger=None)
```

#### Logging Module (`common/logger.py`)
```python
def get_server_logger()
def get_client_logger(client_id)
def setup_colored_logging()
def setup_file_logging(logger, log_file)
```

## Communication Architecture

### Kafka Integration

The system uses Apache Kafka for asynchronous, reliable communication between components.

#### Topic Structure
```
global_model (1 partition)
├── Server publishes global model weights
├── All clients subscribe to receive updates
└── Retention: 1 hour

model_updates (3 partitions)
├── Clients publish their trained model updates
├── Server subscribes to collect updates
└── Retention: 24 hours
```

#### Message Flow
```
1. Server → global_model → All Clients
   ├── Global model weights (serialized)
   ├── Model metadata
   └── Timestamp information

2. Clients → model_updates → Server
   ├── Updated model weights (serialized)
   ├── Client identification
   ├── Training metrics
   └── Timestamp information
```

### Serialization Protocol

#### Weight Serialization Format
```python
{
    "weights": [numpy_array_1, numpy_array_2, ...],
    "metadata": {
        "num_layers": int,
        "shapes": [(shape1), (shape2), ...],
        "dtypes": [dtype1, dtype2, ...],
        "timestamp": float,
        "compression": "gzip"
    },
    "checksum": "sha256_hash"
}
```

#### Binary Protocol
1. **Header**: Magic bytes + version + metadata length
2. **Metadata**: JSON-encoded metadata
3. **Payload**: Compressed serialized weights
4. **Footer**: Checksum for integrity verification

## Data Flow Architecture

### Training Data Flow
```
CIFAR-10 Dataset
       ↓
┌─────────────────┐
│ Data Loading    │ ← load_cifar10_data()
│ & Preprocessing │
└─────────────────┘
       ↓
┌─────────────────┐
│ Data Partitioning│ ← Per-client subsets
└─────────────────┘
       ↓
┌─────────────────┐
│ Local Training  │ ← Client-specific training
└─────────────────┘
       ↓
┌─────────────────┐
│ Model Updates   │ ← Weight differences
└─────────────────┘
```

### Aggregation Data Flow
```
Client Updates (N clients)
       ↓
┌─────────────────┐
│ Update Collection│ ← Server receives via Kafka
└─────────────────┘
       ↓
┌─────────────────┐
│ Weight Validation│ ← Compatibility checks
└─────────────────┘
       ↓
┌─────────────────┐
│ Federated       │ ← FedAvg algorithm
│ Averaging       │
└─────────────────┘
       ↓
┌─────────────────┐
│ Global Model    │ ← Updated global weights
│ Update          │
└─────────────────┘
```

## Deployment Architecture

### Docker Deployment
```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    
  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on: [zookeeper]
    
  federated-server:
    build: 
      context: .
      dockerfile: Dockerfile.server
    depends_on: [kafka]
    
  federated-client-1:
    build:
      context: .
      dockerfile: Dockerfile.client
    depends_on: [kafka]
    environment:
      - CLIENT_ID=1
      
  federated-client-2:
    build:
      context: .
      dockerfile: Dockerfile.client
    depends_on: [kafka]
    environment:
      - CLIENT_ID=2
```

### Local Development Architecture
```
Local Machine
├── Python Virtual Environment
├── Local Kafka Instance (Docker or Manual)
├── Server Process
├── Multiple Client Processes
└── Shared File System (logs, plots, data)
```

## Scalability Architecture

### Horizontal Scaling

#### Client Scaling
- **Dynamic Client Addition**: Clients can join/leave at runtime
- **Load Distribution**: Kafka partitioning for load balancing
- **Resource Isolation**: Independent client processes

#### Server Scaling (Future)
- **Multi-Server Deployment**: Distributed server architecture
- **Load Balancing**: Round-robin client assignment
- **State Synchronization**: Shared model state management

### Vertical Scaling

#### Memory Optimization
- **Lazy Loading**: Load data on-demand
- **Memory Mapping**: Efficient data access
- **Garbage Collection**: Proactive memory management

#### CPU Optimization
- **Parallel Processing**: Multi-threaded operations
- **Vectorization**: NumPy optimized operations
- **Batch Processing**: Efficient batch operations

## Security Architecture

### Data Security
- **Local Data Privacy**: Data never leaves client devices
- **In-Transit Encryption**: Kafka SSL/TLS support
- **Model Obfuscation**: Weight perturbation techniques

### Communication Security
- **Authentication**: Kafka SASL authentication
- **Authorization**: Topic-level access control
- **Integrity Verification**: Checksum validation

### System Security
- **Container Isolation**: Docker security boundaries
- **Resource Limits**: Memory and CPU constraints
- **Network Segmentation**: Isolated communication channels

## Monitoring Architecture

### Metrics Collection
```python
# Server Metrics
{
    "aggregation_times": [],        # Aggregation timestamps
    "update_counts": [],           # Updates per aggregation
    "global_accuracy": [],         # Global model accuracy
    "weight_history": [],          # Model evolution tracking
    "client_updates": {}           # Client similarity analysis
}

# Client Metrics
{
    "train_accuracy": [],          # Training accuracy per round
    "test_accuracy": [],           # Test accuracy per round
    "train_loss": [],              # Training loss per round
    "training_times": []           # Time spent training
}
```

### Visualization Pipeline
1. **Metrics Collection**: Real-time performance tracking
2. **Data Aggregation**: Statistical analysis and summarization
3. **Plot Generation**: Automated visualization creation
4. **Report Generation**: Comprehensive performance reports

## Performance Architecture

### Optimization Strategies

#### Model Optimization
- **Architecture Selection**: LeNet vs CNN trade-offs
- **Parameter Reduction**: Efficient model architectures
- **Quantization**: Reduced precision weights

#### Communication Optimization
- **Compression**: Weight compression techniques
- **Batching**: Message batching for efficiency
- **Async Processing**: Non-blocking operations

#### Computation Optimization
- **CPU/GPU Selection**: Hardware-appropriate processing
- **Memory Management**: Efficient memory usage
- **Parallel Processing**: Multi-core utilization

## Future Architecture Considerations

### Planned Enhancements
1. **Differential Privacy**: Privacy-preserving aggregation
2. **Secure Aggregation**: Cryptographic protection
3. **Client Selection**: Intelligent client sampling
4. **Adaptive Learning**: Dynamic parameter adjustment
5. **Cross-Platform Support**: Mobile and edge devices

### Architectural Evolution
- **Microservices**: Service-oriented architecture
- **Cloud Native**: Kubernetes deployment
- **Event Sourcing**: Complete audit trail
- **CQRS Pattern**: Command-query separation

This architecture provides a solid foundation for scalable, reliable federated learning while maintaining flexibility for future enhancements.
