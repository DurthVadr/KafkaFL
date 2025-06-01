# Kafka Integration

This document details the Apache Kafka integration in the federated learning system, covering topics, message patterns, and optimization strategies.

## Overview

Apache Kafka serves as the distributed messaging backbone for the federated learning system, enabling:
- **Asynchronous Communication**: Non-blocking message exchange
- **Scalability**: Support for numerous clients
- **Reliability**: Message persistence and delivery guarantees
- **Fault Tolerance**: Automatic recovery and replication

## Topic Architecture

### Topic Design

The system uses two primary Kafka topics:

#### 1. Global Model Topic (`global_model`)
- **Purpose**: Server-to-client model distribution
- **Pattern**: One-to-many broadcast
- **Partitions**: 1 (ensures ordering)
- **Replication Factor**: 1 (configurable for production)
- **Retention**: 1 hour (configurable)

```python
GLOBAL_MODEL_TOPIC_CONFIG = {
    "name": "global_model",
    "partitions": 1,
    "replication_factor": 1,
    "config": {
        "retention.ms": "3600000",  # 1 hour
        "cleanup.policy": "delete",
        "max.message.bytes": "10485760"  # 10MB
    }
}
```

#### 2. Model Updates Topic (`model_updates`)
- **Purpose**: Client-to-server model updates
- **Pattern**: Many-to-one collection
- **Partitions**: 3 (load distribution)
- **Replication Factor**: 1 (configurable for production)
- **Retention**: 24 hours (configurable)

```python
MODEL_UPDATES_TOPIC_CONFIG = {
    "name": "model_updates",
    "partitions": 3,
    "replication_factor": 1,
    "config": {
        "retention.ms": "86400000",  # 24 hours
        "cleanup.policy": "delete",
        "max.message.bytes": "10485760"  # 10MB
    }
}
```

### Topic Creation

Topics are automatically created with default settings, but can be pre-created for production:

```bash
# Create global model topic
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic global_model \
  --partitions 1 \
  --replication-factor 1

# Create model updates topic
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic model_updates \
  --partitions 3 \
  --replication-factor 1
```

## Message Patterns

### Server Message Pattern

#### Global Model Broadcasting
```python
# Server publishes global model
def send_global_model(self):
    # Serialize model weights
    serialized_model = serialize_weights(self.global_model)
    
    # Create message with metadata
    message = {
        "timestamp": time.time(),
        "model_version": self.model_version,
        "payload": serialized_model
    }
    
    # Send to all clients via global_model topic
    self.producer.send("global_model", message)
```

#### Update Collection
```python
# Server collects client updates
def receive_model_updates(self, max_updates=3, timeout_ms=60000):
    messages = []
    
    # Poll for messages with timeout
    msg_pack = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_updates)
    
    for topic_partition, msgs in msg_pack.items():
        for msg in msgs:
            # Deserialize and validate update
            update = deserialize_weights(msg.value)
            if update:
                messages.append(update)
    
    return messages
```

### Client Message Pattern

#### Model Reception
```python
# Client receives global model
def receive_global_model(self):
    # Poll for latest global model
    msg_pack = self.consumer.poll(timeout_ms=60000, max_records=1)
    
    for topic_partition, msgs in msg_pack.items():
        if msgs:
            # Get the latest message
            latest_msg = msgs[-1]
            return deserialize_weights(latest_msg.value)
    
    return None
```

#### Update Transmission
```python
# Client sends model update
def send_model_update(self, weights):
    # Serialize updated weights
    serialized_update = serialize_weights(weights)
    
    # Create update message
    message = {
        "client_id": self.client_id,
        "timestamp": time.time(),
        "round_number": self.round_number,
        "payload": serialized_update
    }
    
    # Send update to server
    self.producer.send("model_updates", message)
```

## Message Format

### Message Structure

All messages follow a standardized format:

```python
{
    "header": {
        "version": "1.0",
        "type": "model_weights",
        "timestamp": 1640995200.0,
        "sender": "client_1",
        "message_id": "uuid4_string"
    },
    "metadata": {
        "model_version": 5,
        "num_layers": 8,
        "total_params": 83126,
        "compression": "gzip",
        "checksum": "sha256_hash"
    },
    "payload": "base64_encoded_serialized_weights"
}
```

### Serialization Protocol

#### Weight Serialization
```python
def serialize_weights(weights, logger=None):
    """Serialize model weights for Kafka transmission."""
    try:
        # Convert TensorFlow weights to numpy arrays
        numpy_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in weights]
        
        # Create metadata
        metadata = {
            "num_layers": len(numpy_weights),
            "shapes": [w.shape for w in numpy_weights],
            "dtypes": [str(w.dtype) for w in numpy_weights],
            "timestamp": time.time()
        }
        
        # Serialize with pickle
        weight_bytes = pickle.dumps(numpy_weights)
        
        # Compress data
        compressed_data = gzip.compress(weight_bytes)
        
        # Calculate checksum
        checksum = hashlib.sha256(compressed_data).hexdigest()
        
        # Create final message
        message = {
            "metadata": metadata,
            "payload": base64.b64encode(compressed_data).decode('utf-8'),
            "checksum": checksum
        }
        
        return json.dumps(message).encode('utf-8')
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to serialize weights: {e}")
        return None
```

#### Weight Deserialization
```python
def deserialize_weights(data, logger=None):
    """Deserialize model weights from Kafka message."""
    try:
        # Parse JSON message
        message = json.loads(data.decode('utf-8'))
        
        # Extract components
        metadata = message["metadata"]
        payload = message["payload"]
        checksum = message["checksum"]
        
        # Decode and decompress
        compressed_data = base64.b64decode(payload)
        
        # Verify checksum
        if hashlib.sha256(compressed_data).hexdigest() != checksum:
            raise ValueError("Checksum verification failed")
        
        # Decompress and deserialize
        weight_bytes = gzip.decompress(compressed_data)
        weights = pickle.loads(weight_bytes)
        
        return weights
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to deserialize weights: {e}")
        return None
```

## Producer Configuration

### Server Producer Settings
```python
SERVER_PRODUCER_CONFIG = {
    'bootstrap_servers': ['localhost:9094'],
    'client_id': 'federated_server',
    'acks': 1,                          # Wait for leader acknowledgment
    'retries': 3,                       # Retry failed sends
    'batch_size': 16384,                # Batch size in bytes
    'linger_ms': 100,                   # Wait time for batching
    'buffer_memory': 33554432,          # Total memory for buffering
    'compression_type': 'gzip',         # Message compression
    'max_request_size': 10485760,       # 10MB max message size
    'request_timeout_ms': 30000,        # Request timeout
    'delivery_timeout_ms': 120000,      # Total delivery timeout
    'enable_idempotence': True,         # Prevent duplicate messages
    'max_in_flight_requests_per_connection': 1  # Ensure ordering
}
```

### Client Producer Settings
```python
CLIENT_PRODUCER_CONFIG = {
    'bootstrap_servers': ['localhost:9094'],
    'client_id': f'federated_client_{client_id}',
    'acks': 1,                          # Wait for leader acknowledgment
    'retries': 3,                       # Retry failed sends
    'batch_size': 8192,                 # Smaller batch size
    'linger_ms': 50,                    # Shorter wait time
    'compression_type': 'gzip',         # Message compression
    'max_request_size': 10485760,       # 10MB max message size
    'request_timeout_ms': 30000,        # Request timeout
    'enable_idempotence': False         # Allow duplicates for performance
}
```

## Consumer Configuration

### Server Consumer Settings
```python
SERVER_CONSUMER_CONFIG = {
    'bootstrap_servers': ['localhost:9094'],
    'group_id': 'federated_server',
    'client_id': 'federated_server_consumer',
    'auto_offset_reset': 'latest',      # Start from latest messages
    'enable_auto_commit': True,         # Automatic offset commits
    'auto_commit_interval_ms': 1000,    # Commit interval
    'max_poll_records': 10,             # Max records per poll
    'max_poll_interval_ms': 300000,     # Max time between polls
    'session_timeout_ms': 30000,        # Session timeout
    'heartbeat_interval_ms': 3000,      # Heartbeat interval
    'fetch_min_bytes': 1,               # Minimum fetch size
    'fetch_max_wait_ms': 500,           # Maximum wait time
    'max_partition_fetch_bytes': 1048576 # 1MB max per partition
}
```

### Client Consumer Settings
```python
CLIENT_CONSUMER_CONFIG = {
    'bootstrap_servers': ['localhost:9094'],
    'group_id': f'federated_client_{client_id}',
    'client_id': f'federated_client_{client_id}_consumer',
    'auto_offset_reset': 'latest',      # Start from latest messages
    'enable_auto_commit': True,         # Automatic offset commits
    'auto_commit_interval_ms': 1000,    # Commit interval
    'max_poll_records': 1,              # Only need latest model
    'max_poll_interval_ms': 300000,     # Max time between polls
    'session_timeout_ms': 30000,        # Session timeout
    'heartbeat_interval_ms': 3000       # Heartbeat interval
}
```

## Error Handling and Resilience

### Connection Management
```python
def create_producer_with_retry(bootstrap_servers, config, max_retries=3):
    """Create Kafka producer with retry logic."""
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                **config
            )
            # Test connection
            producer.bootstrap_connected()
            return producer
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### Message Delivery Guarantees
```python
def send_message_with_callback(producer, topic, message, logger=None):
    """Send message with delivery confirmation."""
    try:
        future = producer.send(topic, message)
        
        # Add success/error callbacks
        future.add_callback(lambda metadata: 
            logger.info(f"Message sent to {metadata.topic}:{metadata.partition}:{metadata.offset}")
            if logger else None
        )
        
        future.add_errback(lambda exception:
            logger.error(f"Failed to send message: {exception}")
            if logger else None
        )
        
        # Wait for delivery (optional, for synchronous behavior)
        record_metadata = future.get(timeout=10)
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Message send failed: {e}")
        return False
```

### Consumer Resilience
```python
def consume_with_retry(consumer, timeout_ms=30000, max_retries=3):
    """Consume messages with retry logic."""
    for attempt in range(max_retries):
        try:
            msg_pack = consumer.poll(timeout_ms=timeout_ms)
            return msg_pack
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Brief pause before retry
    
    return {}
```

## Performance Optimization

### Batch Processing
```python
def send_batch_updates(producer, topic, updates, batch_size=10):
    """Send multiple updates in batches for better throughput."""
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i + batch_size]
        
        for update in batch:
            producer.send(topic, update)
        
        # Flush batch
        producer.flush()
```

### Compression Strategies
```python
COMPRESSION_SETTINGS = {
    'gzip': {
        'compression_type': 'gzip',
        'pros': 'High compression ratio',
        'cons': 'CPU intensive',
        'use_case': 'Large models, slow networks'
    },
    'snappy': {
        'compression_type': 'snappy',
        'pros': 'Fast compression/decompression',
        'cons': 'Lower compression ratio',
        'use_case': 'Real-time applications'
    },
    'lz4': {
        'compression_type': 'lz4',
        'pros': 'Very fast, good ratio',
        'cons': 'Moderate compression',
        'use_case': 'Balanced performance'
    }
}
```

### Memory Management
```python
def optimize_kafka_memory():
    """Configure Kafka for memory-efficient operation."""
    config = {
        'buffer_memory': 16777216,      # 16MB buffer
        'batch_size': 8192,             # 8KB batches
        'max_request_size': 5242880,    # 5MB max request
        'receive_buffer_bytes': 65536,   # 64KB receive buffer
        'send_buffer_bytes': 131072      # 128KB send buffer
    }
    return config
```

## Monitoring and Debugging

### Kafka Metrics
```python
def collect_kafka_metrics(producer, consumer, logger):
    """Collect and log Kafka performance metrics."""
    # Producer metrics
    producer_metrics = producer.metrics()
    
    # Key metrics to monitor
    important_metrics = [
        'record-send-rate',
        'record-error-rate',
        'batch-size-avg',
        'compression-rate-avg'
    ]
    
    for metric_name in important_metrics:
        for metric_key, metric_value in producer_metrics.items():
            if metric_name in str(metric_key):
                logger.info(f"Producer {metric_name}: {metric_value.value}")
```

### Topic Monitoring
```bash
# Monitor topic lag
kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --group federated_server --describe

# Monitor topic throughput
kafka-run-class.sh kafka.tools.ConsumerPerformance \
  --topic model_updates \
  --broker-list localhost:9092 \
  --messages 1000
```

### Debug Tools
```python
def debug_kafka_connection(bootstrap_servers):
    """Debug Kafka connectivity issues."""
    try:
        # Test basic connection
        client = KafkaClient(bootstrap_servers=bootstrap_servers)
        
        # Check cluster metadata
        metadata = client.cluster
        print(f"Cluster ID: {metadata.cluster_id}")
        print(f"Controller: {metadata.controller}")
        print(f"Brokers: {list(metadata.brokers.keys())}")
        
        # Check topic metadata
        for topic in metadata.topics:
            print(f"Topic: {topic}")
            for partition in metadata.partitions_for_topic(topic):
                print(f"  Partition {partition}")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False
```

## Security Configuration

### SSL/TLS Configuration
```python
SSL_CONFIG = {
    'security_protocol': 'SSL',
    'ssl_cafile': '/path/to/ca-cert',
    'ssl_certfile': '/path/to/client-cert',
    'ssl_keyfile': '/path/to/client-key',
    'ssl_password': 'keystore_password',
    'ssl_check_hostname': True
}
```

### SASL Authentication
```python
SASL_CONFIG = {
    'security_protocol': 'SASL_PLAINTEXT',
    'sasl_mechanism': 'PLAIN',
    'sasl_plain_username': 'username',
    'sasl_plain_password': 'password'
}
```

This Kafka integration provides a robust, scalable foundation for federated learning communication while maintaining high performance and reliability.
