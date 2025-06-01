# Configuration Guide

This guide covers all configuration options available in the federated learning system.

## Configuration Overview

The system supports multiple configuration methods:
1. Environment variables
2. Command-line arguments
3. Configuration files
4. Docker Compose settings

## Environment Variables

### Kafka Configuration

```bash
# Kafka broker address
BOOTSTRAP_SERVERS=localhost:9094

# Alternative for Docker environment
BOOTSTRAP_SERVERS=kafka:9092

# Kafka connection timeout
KAFKA_TIMEOUT_MS=30000

# Kafka producer settings
KAFKA_BATCH_SIZE=16384
KAFKA_LINGER_MS=100
```

### Application Configuration

```bash
# Client identification
CLIENT_ID=auto                          # Auto-generate or specify

# Runtime duration
DURATION_MINUTES=60                      # Total runtime in minutes

# Server aggregation settings
AGGREGATION_INTERVAL_SECONDS=60          # Time between aggregations
MIN_UPDATES_PER_AGGREGATION=1           # Minimum updates before aggregation

# Client training settings
TRAINING_INTERVAL_SECONDS=120            # Time between client training cycles
NUM_CLIENTS=3                           # Number of clients to start

# Data configuration
REDUCED_DATA_SIZE=0                      # Use full dataset (0) or reduced (1)
TRAIN_SUBSET_SIZE=16000                  # Max training samples per client
TEST_SUBSET_SIZE=1000                    # Max test samples for evaluation
```

### TensorFlow Configuration

```bash
# Logging level (0=all, 1=info, 2=warnings, 3=errors only)
TF_CPP_MIN_LOG_LEVEL=2

# GPU configuration
CUDA_VISIBLE_DEVICES=-1                  # Force CPU-only (-1) or specify GPU

# Memory management
TF_FORCE_GPU_ALLOW_GROWTH=true          # Allow gradual GPU memory allocation
TF_MEMORY_LIMIT=1024                     # Limit GPU memory (MB)
```

### Logging Configuration

```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Log output
LOG_TO_FILE=true                         # Enable file logging
LOG_TO_CONSOLE=true                      # Enable console logging
LOG_COLOR=true                           # Enable colored console output

# Log file settings
LOG_ROTATION=true                        # Enable log rotation
LOG_MAX_SIZE=10MB                        # Maximum log file size
LOG_BACKUP_COUNT=5                       # Number of backup log files
```

## Command-Line Configuration

### Server Configuration

```bash
python server.py [OPTIONS]

# Available options:
--bootstrap-servers TEXT    # Kafka bootstrap servers
--duration INTEGER          # Runtime duration in minutes
--aggregation-interval INTEGER  # Aggregation interval in seconds
--min-updates INTEGER       # Minimum updates per aggregation
--log-level TEXT           # Logging level
```

### Client Configuration

```bash
python client.py [OPTIONS]

# Available options:
--bootstrap-servers TEXT    # Kafka bootstrap servers
--client-id INTEGER         # Client identifier
--duration INTEGER          # Runtime duration in minutes
--training-interval INTEGER # Training interval in seconds
--log-level TEXT           # Logging level
```

### Local Runner Configuration

```bash
python run_local_kafka_no_docker.py [OPTIONS]

# Available options:
--duration INTEGER          # Duration in minutes (default: 30)
--aggregation-interval INTEGER  # Server aggregation interval (default: 30)
--min-updates INTEGER       # Minimum updates required (default: 1)
--training-interval INTEGER # Client training interval (default: 60)
--num-clients INTEGER       # Number of clients (default: 3)
--reduced-data BOOLEAN      # Use reduced dataset (default: True)
```

## Model Configuration

### Model Architecture Selection

```python
# In common/model.py
MODEL_TYPE = "lenet"  # Options: "lenet", "cnn"

# LeNet configuration
LENET_CONFIG = {
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "dropout_rate": 0.2
}

# CNN configuration
CNN_CONFIG = {
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "filters": [32, 64, 128],
    "dropout_rate": 0.5
}
```

### Training Parameters

```python
# Training configuration
TRAINING_CONFIG = {
    "epochs": 5,                    # Epochs per training round
    "batch_size": 32,               # Training batch size
    "learning_rate": 0.001,         # Learning rate
    "validation_split": 0.2,        # Validation data percentage
    "shuffle": True,                # Shuffle training data
    "verbose": 0                    # Training verbosity
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "batch_size": 32,               # Evaluation batch size
    "verbose": 0                    # Evaluation verbosity
}
```

## Kafka Topic Configuration

### Topic Settings

```bash
# Topic names
MODEL_TOPIC=global_model              # Server-to-client model distribution
UPDATE_TOPIC=model_updates            # Client-to-server updates

# Topic configuration
TOPIC_PARTITIONS=3                    # Number of partitions per topic
TOPIC_REPLICATION_FACTOR=1            # Replication factor
TOPIC_RETENTION_MS=86400000           # Retention time (24 hours)
```

### Message Configuration

```bash
# Message size limits
MAX_MESSAGE_SIZE=10485760             # 10MB maximum message size
COMPRESSION_TYPE=gzip                 # Message compression (gzip, snappy, lz4)

# Producer settings
PRODUCER_ACKS=1                       # Acknowledgment level
PRODUCER_RETRIES=3                    # Number of retries
PRODUCER_TIMEOUT_MS=30000             # Producer timeout

# Consumer settings
CONSUMER_TIMEOUT_MS=60000             # Consumer timeout
CONSUMER_AUTO_OFFSET_RESET=latest     # Offset reset strategy
```

## Docker Configuration

### Docker Compose Environment

```yaml
# docker-compose.yml environment variables
environment:
  # Kafka settings
  - BOOTSTRAP_SERVERS=kafka:9092
  
  # Application settings
  - DURATION_MINUTES=30
  - AGGREGATION_INTERVAL_SECONDS=30
  - TRAINING_INTERVAL_SECONDS=60
  - REDUCED_DATA_SIZE=1
  
  # Resource limits
  - TF_CPP_MIN_LOG_LEVEL=2
  - CUDA_VISIBLE_DEVICES=-1
```

### Resource Limits

```yaml
# Resource constraints in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## Performance Configuration

### Memory Optimization

```bash
# Reduce memory usage
REDUCED_DATA_SIZE=1                   # Use smaller dataset
TRAIN_SUBSET_SIZE=8000                # Reduce training data
TEST_SUBSET_SIZE=500                  # Reduce test data

# TensorFlow memory settings
TF_ENABLE_ONEDNN_OPTS=0              # Disable oneDNN optimizations
TF_DISABLE_MKL=1                     # Disable Intel MKL
```

### CPU Optimization

```bash
# Thread configuration
TF_INTER_OP_PARALLELISM_THREADS=1     # Inter-op parallelism
TF_INTRA_OP_PARALLELISM_THREADS=1     # Intra-op parallelism

# Process settings
OMP_NUM_THREADS=1                     # OpenMP threads
OPENBLAS_NUM_THREADS=1                # OpenBLAS threads
```

### Network Optimization

```bash
# Kafka network settings
KAFKA_SOCKET_SEND_BUFFER_BYTES=102400
KAFKA_SOCKET_RECEIVE_BUFFER_BYTES=102400
KAFKA_SEND_BUFFER_BYTES=131072
KAFKA_RECEIVE_BUFFER_BYTES=65536

# Connection pooling
KAFKA_CONNECTIONS_MAX_IDLE_MS=540000
KAFKA_MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION=5
```

## Monitoring Configuration

### Metrics Collection

```bash
# Enable/disable metrics
ENABLE_METRICS=true                   # Collect performance metrics
METRICS_INTERVAL=10                   # Metrics collection interval (seconds)

# Visualization settings
GENERATE_PLOTS=true                   # Generate performance plots
PLOT_FORMAT=png                       # Plot format (png, pdf, svg)
PLOT_DPI=300                         # Plot resolution
```

### Logging Configuration

```bash
# Log file locations
LOG_DIR=logs                         # Log directory
SERVER_LOG_FILE=server.log           # Server log file
CLIENT_LOG_FILE=client_{client_id}.log  # Client log file pattern

# Log format
LOG_FORMAT='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT='%Y-%m-%d %H:%M:%S'
```

## Configuration Examples

### Development Environment

```bash
# .env file for development
BOOTSTRAP_SERVERS=localhost:9094
DURATION_MINUTES=10
AGGREGATION_INTERVAL_SECONDS=20
TRAINING_INTERVAL_SECONDS=30
NUM_CLIENTS=2
REDUCED_DATA_SIZE=1
TF_CPP_MIN_LOG_LEVEL=2
LOG_LEVEL=DEBUG
```

### Production Environment

```bash
# .env file for production
BOOTSTRAP_SERVERS=kafka-cluster:9092
DURATION_MINUTES=360
AGGREGATION_INTERVAL_SECONDS=300
TRAINING_INTERVAL_SECONDS=600
NUM_CLIENTS=10
REDUCED_DATA_SIZE=0
TF_CPP_MIN_LOG_LEVEL=1
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

### High-Performance Environment

```bash
# .env file for high-performance setup
BOOTSTRAP_SERVERS=kafka-cluster:9092
DURATION_MINUTES=180
AGGREGATION_INTERVAL_SECONDS=60
TRAINING_INTERVAL_SECONDS=120
NUM_CLIENTS=20
TRAIN_SUBSET_SIZE=32000
TEST_SUBSET_SIZE=2000
KAFKA_BATCH_SIZE=32768
COMPRESSION_TYPE=lz4
```

## Configuration Validation

### Validation Scripts

```bash
# Validate configuration
python -c "
import os
from common.logger import get_server_logger
logger = get_server_logger()
logger.info('Configuration validation passed')
"

# Check Kafka connectivity
python -c "
from common.kafka_utils import create_producer
producer = create_producer(os.getenv('BOOTSTRAP_SERVERS', 'localhost:9094'))
print('Kafka connection:', 'OK' if producer else 'FAILED')
"
```

### Configuration Testing

```bash
# Test with minimal configuration
python run_local_kafka_no_docker.py --duration 1 --num-clients 1

# Test with custom configuration
export REDUCED_DATA_SIZE=1
export TF_CPP_MIN_LOG_LEVEL=3
python server.py &
python client.py
```

## Best Practices

### Security Considerations

```bash
# Kafka security (for production)
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_SASL_USERNAME=your_username
KAFKA_SASL_PASSWORD=your_password
```

### Resource Management

```bash
# Monitor resource usage
MEMORY_LIMIT=4096                    # Memory limit in MB
CPU_LIMIT=2.0                       # CPU limit (cores)
DISK_SPACE_LIMIT=10240              # Disk space limit in MB
```

### Error Handling

```bash
# Error handling configuration
MAX_RETRIES=3                       # Maximum retry attempts
RETRY_DELAY=5                       # Delay between retries (seconds)
FAIL_ON_ERROR=false                 # Continue on non-critical errors
```

For more advanced configuration options, see the [API Reference](api_reference.md) and [System Architecture](system_architecture.md) documentation.
