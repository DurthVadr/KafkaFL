# Server Implementation

This document provides a detailed overview of the federated learning server implementation, including architecture, algorithms, and operational details.

## Overview

The `FederatedServer` class serves as the central coordinator for the federated learning system. It manages the global model, orchestrates client interactions, and performs model aggregation using the FedAvg algorithm.

## Class Architecture

### Core Structure

```python
class FederatedServer:
    """Server for federated learning with Kafka-based communication."""
    
    def __init__(self, bootstrap_servers, model_topic, update_topic):
        """Initialize server with Kafka configuration."""
        
    def _initialize_global_model(self):
        """Create initial global model weights."""
        
    def _connect_to_kafka(self):
        """Establish Kafka producer and consumer connections."""
        
    def send_global_model(self):
        """Broadcast global model to all clients."""
        
    def receive_model_updates(self, max_updates=3, timeout_ms=60000):
        """Collect model updates from clients."""
        
    def aggregate_model_updates(self, updates):
        """Perform federated averaging on client updates."""
        
    def start(self, duration_minutes=60, aggregation_interval_seconds=60, min_updates_per_aggregation=1):
        """Start the federated learning process."""
        
    def generate_visualizations(self):
        """Create performance visualizations and reports."""
        
    def close(self):
        """Clean up resources and connections."""
```

### Initialization Process

#### 1. Server Setup
```python
def __init__(self, bootstrap_servers, model_topic, update_topic):
    # Initialize logger
    self.logger = get_server_logger()
    
    # Configure Kafka settings
    self.bootstrap_servers = bootstrap_servers
    self.model_topic = model_topic
    self.update_topic = update_topic
    
    # Initialize global model
    self.global_model = self._initialize_global_model()
    
    # Setup metrics tracking
    self.metrics = {
        'aggregation_times': [],     # Timestamps of aggregations
        'update_counts': [],         # Number of updates per aggregation
        'global_accuracy': [],       # Global model accuracy over time
        'weight_history': [],        # Evolution of model weights
        'client_updates': {}         # Client update similarity analysis
    }
    
    # Establish Kafka connections
    self._connect_to_kafka()
```

#### 2. Global Model Initialization
```python
def _initialize_global_model(self):
    """Initialize the global model with LeNet architecture."""
    self.logger.info("Initializing global model (LeNet)")
    
    if TENSORFLOW_AVAILABLE:
        # Create LeNet model
        model = create_lenet_model()
        if model is not None:
            weights = model.get_weights()
            self.logger.info(f"Initialized LeNet model with {len(weights)} layers")
            
            # Log weight shapes for debugging
            for i, w in enumerate(weights):
                self.logger.info(f"Layer {i} shape: {w.shape}")
            
            return weights
    
    # Fallback to random weights
    self.logger.warning("Using random weights for LeNet model")
    weights = get_random_weights(model_type="lenet")
    return weights
```

#### 3. Kafka Connection Setup
```python
def _connect_to_kafka(self):
    """Establish Kafka producer and consumer connections."""
    self.logger.info(f"Connecting to Kafka at {self.bootstrap_servers}")
    
    # Create producer for sending global models
    self.producer = create_producer(
        bootstrap_servers=self.bootstrap_servers,
        logger=self.logger
    )
    
    # Create consumer for receiving client updates
    self.consumer = create_consumer(
        bootstrap_servers=self.bootstrap_servers,
        group_id="federated_server",
        topics=[self.update_topic],
        logger=self.logger
    )
    
    # Verify connections
    if self.producer is not None and self.consumer is not None:
        self.logger.info("Successfully connected to Kafka")
        return True
    else:
        self.logger.error("Failed to connect to Kafka")
        return False
```

## Core Operations

### 1. Global Model Distribution

```python
def send_global_model(self):
    """Send the global model to all clients via Kafka."""
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
```

### 2. Client Update Collection

```python
def receive_model_updates(self, max_updates=3, timeout_ms=60000):
    """Receive model updates from clients with configurable timeout."""
    
    # Log based on timeout duration (for async vs sync operation)
    if timeout_ms < 5000:
        self.logger.debug(f"Polling for model updates (max: {max_updates})")
    else:
        self.logger.info(f"Waiting for model updates from clients on topic {self.update_topic}")
    
    # Receive messages from the update topic
    messages = receive_messages(
        consumer=self.consumer,
        timeout_ms=timeout_ms,
        max_messages=max_updates,
        logger=self.logger
    )
    
    if not messages:
        if timeout_ms >= 5000:
            self.logger.warning("No model updates received from clients")
        return []
    
    # Deserialize model updates
    updates = []
    for i, message in enumerate(messages):
        update = deserialize_weights(message, logger=self.logger)
        if update is not None:
            updates.append(update)
            
            # Store client update for similarity analysis
            client_id = f"client_{int(time.time())}_{i}"
            self.metrics['client_updates'][client_id] = update
    
    if updates:
        self.logger.info(f"Received {len(updates)} valid model updates from clients")
    
    return updates
```

### 3. Federated Averaging Algorithm

```python
def aggregate_model_updates(self, updates):
    """Perform federated averaging on client model updates."""
    
    if not updates:
        self.logger.warning("No updates to aggregate")
        return self.global_model
    
    self.logger.info(f"Aggregating {len(updates)} model updates using FedAvg")
    
    try:
        # Initialize aggregated weights
        aggregated_weights = []
        num_layers = len(updates[0])
        
        # Aggregate each layer separately
        for layer_idx in range(num_layers):
            # Extract weights for this layer from all clients
            layer_weights = [update[layer_idx] for update in updates]
            
            # Check if all weights have the same shape
            if not all(w.shape == layer_weights[0].shape for w in layer_weights):
                self.logger.error(f"Layer {layer_idx} has inconsistent shapes across updates.")
                return self.global_model
            
            # Average the weights (FedAvg algorithm)
            layer_avg = np.mean(layer_weights, axis=0)
            aggregated_weights.append(layer_avg)
            
            # Free memory
            del layer_weights
        
        self.logger.info("Successfully aggregated model updates")
        
        # Run garbage collection after aggregation
        gc.collect()
        
        return aggregated_weights
        
    except Exception as e:
        self.logger.error(f"Error aggregating model updates: {e}")
        return self.global_model
```

## Asynchronous Operation

### Time-Based Aggregation

The server implements asynchronous federated learning with time-based aggregation:

```python
def start(self, duration_minutes=60, aggregation_interval_seconds=60, min_updates_per_aggregation=1):
    """Start asynchronous federated learning with time-based aggregation."""
    
    self.logger.info(f"Starting federated learning server")
    self.logger.info(f"Duration: {duration_minutes}m, Interval: {aggregation_interval_seconds}s")
    
    # Send initial global model
    if not self.send_global_model():
        self.logger.error("Failed to send initial global model. Exiting.")
        return
    
    # Initialize timing variables
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    last_aggregation_time = start_time
    aggregation_count = 0
    total_updates_received = 0
    pending_updates = []
    
    try:
        while time.time() < end_time:
            current_time = time.time()
            
            # Log progress periodically
            if int(current_time) % 60 == 0:
                time_elapsed_minutes = (current_time - start_time) / 60
                time_remaining_minutes = duration_minutes - time_elapsed_minutes
                self.logger.info(f"Server running: {time_elapsed_minutes:.1f}m elapsed, {time_remaining_minutes:.1f}m remaining")
            
            # Check if it's time to aggregate
            time_since_last_aggregation = current_time - last_aggregation_time
            should_aggregate = (time_since_last_aggregation >= aggregation_interval_seconds and 
                               len(pending_updates) >= min_updates_per_aggregation)
            
            if should_aggregate:
                self.logger.info(f"=== Aggregation {aggregation_count + 1} ===")
                
                # Perform aggregation
                self.global_model = self.aggregate_model_updates(pending_updates)
                
                # Send updated global model to clients
                self.send_global_model()
                
                # Track metrics
                self.metrics['aggregation_times'].append(current_time)
                self.metrics['update_counts'].append(len(pending_updates))
                self.metrics['weight_history'].append([np.copy(w) for w in self.global_model])
                
                # Update tracking variables
                last_aggregation_time = current_time
                aggregation_count += 1
                total_updates_received += len(pending_updates)
                pending_updates = []
                
                # Run garbage collection
                gc.collect()
            
            # Receive any available updates (non-blocking)
            new_updates = self.receive_model_updates(max_updates=10, timeout_ms=1000)
            if new_updates:
                pending_updates.extend(new_updates)
            
            # Brief sleep to avoid tight polling
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        self.logger.info("Server interrupted by user")
    
    # Final statistics and cleanup
    self.logger.info(f"Federated learning completed: {aggregation_count} aggregations, {total_updates_received} updates")
    
    # Perform final aggregation if needed
    if pending_updates:
        self.logger.info(f"Performing final aggregation with {len(pending_updates)} updates")
        self.global_model = self.aggregate_model_updates(pending_updates)
        self.send_global_model()
    
    # Generate visualizations
    self.generate_visualizations()
```

## Metrics and Monitoring

### Performance Tracking

The server maintains comprehensive metrics for monitoring and analysis:

```python
def track_aggregation_metrics(self, updates, aggregation_time):
    """Track metrics for each aggregation round."""
    
    # Basic metrics
    self.metrics['aggregation_times'].append(aggregation_time)
    self.metrics['update_counts'].append(len(updates))
    
    # Model evolution tracking
    self.metrics['weight_history'].append([np.copy(w) for w in self.global_model])
    
    # Client similarity analysis
    if len(updates) > 1:
        similarities = self.calculate_client_similarities(updates)
        self.logger.info(f"Client similarity metrics: {similarities}")
    
    # Memory usage tracking
    import psutil
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    self.logger.debug(f"Memory usage: {memory_usage:.1f} MB")
```

### Visualization Generation

```python
def generate_visualizations(self):
    """Generate comprehensive visualizations of server performance."""
    
    self.logger.info("Generating server visualizations")
    
    if not self.metrics['aggregation_times']:
        self.logger.warning("No metrics available for visualization")
        return
    
    try:
        # 1. Aggregation timeline plot
        plot_path = plot_server_aggregations(
            self.metrics['aggregation_times'],
            self.metrics['update_counts'],
            self.logger
        )
        self.logger.info(f"Server aggregation plot saved to {plot_path}")
        
        # 2. Weight distribution analysis
        if self.metrics['weight_history']:
            latest_weights = self.metrics['weight_history'][-1]
            round_num = len(self.metrics['weight_history'])
            
            violin_plot_path = plot_weight_distribution_violin(
                latest_weights,
                round_num,
                logger=self.logger
            )
            self.logger.info(f"Weight distribution plot saved to {violin_plot_path}")
        
        # 3. Convergence visualization
        if len(self.metrics['weight_history']) >= 2:
            for layer_idx in range(min(3, len(self.global_model))):
                convergence_plot_path = plot_convergence_visualization(
                    self.metrics['weight_history'],
                    layer_idx=layer_idx,
                    logger=self.logger
                )
                self.logger.info(f"Convergence plot for layer {layer_idx} saved to {convergence_plot_path}")
        
        # 4. Client similarity heatmap
        if len(self.metrics['client_updates']) >= 2:
            similarity_plot_path = plot_client_similarity_heatmap(
                self.metrics['client_updates'],
                logger=self.logger
            )
            self.logger.info(f"Client similarity heatmap saved to {similarity_plot_path}")
    
    except Exception as e:
        self.logger.error(f"Error generating server visualizations: {e}")
```

## Error Handling and Resilience

### Connection Management

```python
def handle_kafka_errors(self):
    """Handle Kafka connection errors with automatic recovery."""
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # Test connections
            if self.producer is None or self.consumer is None:
                self.logger.warning(f"Kafka connection lost, attempting reconnection (attempt {attempt + 1})")
                self._connect_to_kafka()
            
            # Test producer
            self.producer.bootstrap_connected()
            
            # Test consumer
            self.consumer.topics()
            
            self.logger.info("Kafka connections restored")
            return True
            
        except Exception as e:
            self.logger.error(f"Kafka reconnection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                self.logger.error("All Kafka reconnection attempts failed")
                return False
    
    return False
```

### Resource Management

```python
def manage_resources(self):
    """Monitor and manage system resources."""
    
    import psutil
    
    # Memory monitoring
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > 80:
        self.logger.warning(f"High memory usage: {memory_percent}%")
        gc.collect()  # Force garbage collection
    
    # CPU monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        self.logger.warning(f"High CPU usage: {cpu_percent}%")
        time.sleep(0.1)  # Brief pause to reduce load
    
    # Disk space monitoring
    disk_usage = psutil.disk_usage('.').percent
    if disk_usage > 90:
        self.logger.error(f"Low disk space: {disk_usage}% used")
```

## Configuration and Optimization

### Performance Tuning

```python
def optimize_performance(self):
    """Apply performance optimizations."""
    
    # TensorFlow optimizations
    if TENSORFLOW_AVAILABLE:
        import tensorflow as tf
        tf.config.optimizer.set_jit(True)  # Enable XLA
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Memory optimizations
    gc.set_threshold(700, 10, 10)  # Aggressive garbage collection
    
    # Kafka optimizations
    self.producer.config['linger_ms'] = 100  # Batch messages
    self.producer.config['compression_type'] = 'gzip'  # Compress messages
```

### Scalability Considerations

```python
def scale_for_clients(self, num_clients):
    """Adjust server settings based on expected client count."""
    
    if num_clients <= 5:
        # Small scale: more frequent aggregations
        self.aggregation_interval = 30
        self.min_updates_per_aggregation = max(1, num_clients // 2)
        
    elif num_clients <= 20:
        # Medium scale: balanced approach
        self.aggregation_interval = 60
        self.min_updates_per_aggregation = max(2, num_clients // 3)
        
    else:
        # Large scale: less frequent aggregations
        self.aggregation_interval = 120
        self.min_updates_per_aggregation = max(5, num_clients // 4)
    
    self.logger.info(f"Scaled for {num_clients} clients: interval={self.aggregation_interval}s, min_updates={self.min_updates_per_aggregation}")
```

## Signal Handling and Cleanup

```python
def close(self):
    """Clean up resources and connections."""
    
    self.logger.info("Closing server resources")
    
    # Close Kafka resources
    close_kafka_resources(
        producer=self.producer,
        consumer=self.consumer,
        logger=self.logger
    )
    
    # Clear model to free memory
    self.global_model = None
    
    # Force garbage collection
    gc.collect()
    
    self.logger.info("Server resources closed")

# Global signal handling
server_instance = None

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global server_instance
    
    if server_instance:
        server_instance.logger.info("Received termination signal, shutting down gracefully")
        server_instance.close()
    
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

## Usage Examples

### Basic Server Operation
```python
# Create and start server
server = FederatedServer(
    bootstrap_servers="localhost:9094",
    model_topic="global_model",
    update_topic="model_updates"
)

# Start federated learning
server.start(
    duration_minutes=60,
    aggregation_interval_seconds=30,
    min_updates_per_aggregation=2
)
```

### Advanced Configuration
```python
# Environment-based configuration
bootstrap_servers = os.environ.get("BOOTSTRAP_SERVERS", "localhost:9094")
duration = int(os.environ.get("DURATION_MINUTES", "60"))
interval = int(os.environ.get("AGGREGATION_INTERVAL_SECONDS", "60"))

server = FederatedServer(bootstrap_servers, "global_model", "model_updates")
server.start(duration, interval, 1)
```

This server implementation provides a robust, scalable foundation for federated learning with comprehensive monitoring, error handling, and performance optimization capabilities.
