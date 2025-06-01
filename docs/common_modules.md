# Common Modules

## Overview

The common modules provide shared functionality used across both client and server components in the federated learning system. These modules ensure consistency, reduce code duplication, and provide essential utilities for communication, data processing, and system operations.

## Module Structure

The common modules are organized into several key areas:

```
common/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── neural_networks.py
│   └── utils.py
├── serialization/
│   ├── __init__.py
│   ├── weight_serializer.py
│   └── message_serializer.py
├── communication/
│   ├── __init__.py
│   ├── kafka_client.py
│   └── message_handlers.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   ├── config.py
│   └── metrics.py
└── exceptions/
    ├── __init__.py
    └── custom_exceptions.py
```

## Model Management (`common/models/`)

### Base Model Classes

The base model module provides abstract classes and interfaces for federated learning models:

```python
# common/models/base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseFLModel(ABC, nn.Module):
    """Abstract base class for federated learning models"""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.model_version = "1.0"
        self.parameter_count = 0
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the model"""
        pass
    
    @abstractmethod
    def get_model_info(self):
        """Return model metadata"""
        pass
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_info(self):
        """Get information about model layers"""
        layer_info = []
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters())
                })
        return layer_info
```

### Neural Network Implementations

```python
# common/models/neural_networks.py
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFLModel

class SimpleMLP(BaseFLModel):
    """Simple Multi-Layer Perceptron for federated learning"""
    
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        super().__init__()
        self.model_name = "SimpleMLP"
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.parameter_count = self.count_parameters()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)
    
    def get_model_info(self):
        return {
            'name': self.model_name,
            'version': self.model_version,
            'parameters': self.parameter_count,
            'input_dim': self.network[0].in_features,
            'output_dim': self.network[-1].out_features,
            'hidden_layers': len([m for m in self.network if isinstance(m, nn.Linear)]) - 1
        }

class SimpleCNN(BaseFLModel):
    """Simple Convolutional Neural Network for federated learning"""
    
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.model_name = "SimpleCNN"
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.parameter_count = self.count_parameters()
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def get_model_info(self):
        return {
            'name': self.model_name,
            'version': self.model_version,
            'parameters': self.parameter_count,
            'input_channels': self.conv1.in_channels,
            'num_classes': self.fc2.out_features,
            'architecture': 'CNN'
        }
```

### Model Utilities

```python
# common/models/utils.py
import torch
import hashlib
import json

def create_model_from_config(config):
    """Factory function to create models from configuration"""
    model_type = config.get('type', 'SimpleMLP')
    
    if model_type == 'SimpleMLP':
        return SimpleMLP(
            input_dim=config.get('input_dim', 784),
            hidden_dims=config.get('hidden_dims', [128, 64]),
            output_dim=config.get('output_dim', 10)
        )
    elif model_type == 'SimpleCNN':
        return SimpleCNN(
            num_classes=config.get('num_classes', 10),
            input_channels=config.get('input_channels', 1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_model_hash(model):
    """Generate hash of model architecture"""
    model_str = str(model)
    return hashlib.md5(model_str.encode()).hexdigest()

def compare_model_architectures(model1, model2):
    """Compare two model architectures"""
    return get_model_hash(model1) == get_model_hash(model2)

def save_model_config(model, filepath):
    """Save model configuration to file"""
    config = {
        'type': model.model_name,
        'info': model.get_model_info(),
        'state_dict_keys': list(model.state_dict().keys())
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_model_config(filepath):
    """Load model configuration from file"""
    with open(filepath, 'r') as f:
        return json.load(f)
```

## Serialization (`common/serialization/`)

### Weight Serialization

```python
# common/serialization/weight_serializer.py
import torch
import pickle
import base64
import gzip
from typing import Dict, Any

class WeightSerializer:
    """Handles serialization and deserialization of model weights"""
    
    @staticmethod
    def serialize_weights(state_dict: Dict[str, torch.Tensor], 
                         compression: bool = True) -> str:
        """Serialize model weights to base64 string"""
        # Convert tensors to CPU and detach
        cpu_state_dict = {k: v.cpu().detach() for k, v in state_dict.items()}
        
        # Pickle the state dict
        pickled_data = pickle.dumps(cpu_state_dict)
        
        # Optionally compress
        if compression:
            pickled_data = gzip.compress(pickled_data)
        
        # Encode to base64
        encoded_data = base64.b64encode(pickled_data).decode('utf-8')
        
        return encoded_data
    
    @staticmethod
    def deserialize_weights(encoded_data: str, 
                          compression: bool = True) -> Dict[str, torch.Tensor]:
        """Deserialize base64 string to model weights"""
        # Decode from base64
        pickled_data = base64.b64decode(encoded_data.encode('utf-8'))
        
        # Optionally decompress
        if compression:
            pickled_data = gzip.decompress(pickled_data)
        
        # Unpickle the state dict
        state_dict = pickle.loads(pickled_data)
        
        return state_dict
    
    @staticmethod
    def get_weights_size(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Get size information about weights"""
        total_params = sum(tensor.numel() for tensor in state_dict.values())
        total_bytes = sum(tensor.element_size() * tensor.numel() 
                         for tensor in state_dict.values())
        
        return {
            'total_parameters': total_params,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'layer_count': len(state_dict),
            'layer_info': {k: list(v.shape) for k, v in state_dict.items()}
        }

class DifferentialWeightSerializer:
    """Handles differential weight updates for efficiency"""
    
    @staticmethod
    def compute_weight_diff(old_weights: Dict[str, torch.Tensor],
                           new_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute difference between weight sets"""
        diff = {}
        for key in old_weights.keys():
            if key in new_weights:
                diff[key] = new_weights[key] - old_weights[key]
        return diff
    
    @staticmethod
    def apply_weight_diff(base_weights: Dict[str, torch.Tensor],
                         diff: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply weight difference to base weights"""
        updated_weights = {}
        for key in base_weights.keys():
            if key in diff:
                updated_weights[key] = base_weights[key] + diff[key]
            else:
                updated_weights[key] = base_weights[key].clone()
        return updated_weights
```

### Message Serialization

```python
# common/serialization/message_serializer.py
import json
import time
from typing import Dict, Any
from datetime import datetime

class MessageSerializer:
    """Handles serialization of Kafka messages"""
    
    @staticmethod
    def serialize_message(message_type: str, payload: Dict[str, Any], 
                         sender_id: str = None) -> str:
        """Serialize message to JSON string"""
        message = {
            'type': message_type,
            'payload': payload,
            'sender_id': sender_id,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        return json.dumps(message, default=str)
    
    @staticmethod
    def deserialize_message(message_str: str) -> Dict[str, Any]:
        """Deserialize JSON string to message"""
        try:
            message = json.loads(message_str)
            
            # Validate required fields
            required_fields = ['type', 'payload', 'timestamp']
            for field in required_fields:
                if field not in message:
                    raise ValueError(f"Missing required field: {field}")
            
            return message
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON message: {e}")
    
    @staticmethod
    def create_client_update_message(client_id: str, round_id: int,
                                   weights: str, metrics: Dict[str, Any]) -> str:
        """Create standardized client update message"""
        payload = {
            'client_id': client_id,
            'round_id': round_id,
            'weights': weights,
            'metrics': metrics
        }
        
        return MessageSerializer.serialize_message(
            'client_update', payload, sender_id=client_id
        )
    
    @staticmethod
    def create_server_broadcast_message(round_id: int, weights: str,
                                      config: Dict[str, Any] = None) -> str:
        """Create standardized server broadcast message"""
        payload = {
            'round_id': round_id,
            'weights': weights,
            'config': config or {}
        }
        
        return MessageSerializer.serialize_message(
            'global_update', payload, sender_id='server'
        )
```

## Communication (`common/communication/`)

### Kafka Client Wrapper

```python
# common/communication/kafka_client.py
import asyncio
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaException
import logging
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

class AsyncKafkaClient:
    """Asynchronous Kafka client wrapper"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = None
        self.consumers = {}
        self.running = False
        self.message_handlers = {}
    
    async def start(self):
        """Start Kafka client"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                value_serializer=lambda v: v.encode('utf-8') if isinstance(v, str) else v,
                **self.config.get('producer_config', {})
            )
            
            self.running = True
            logger.info("Kafka client started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka client: {e}")
            raise
    
    async def stop(self):
        """Stop Kafka client"""
        self.running = False
        
        if self.producer:
            self.producer.close()
        
        for consumer in self.consumers.values():
            consumer.close()
        
        logger.info("Kafka client stopped")
    
    async def send_message(self, topic: str, message: str, key: str = None):
        """Send message to Kafka topic"""
        if not self.producer:
            raise RuntimeError("Producer not initialized")
        
        try:
            future = self.producer.send(topic, value=message, key=key)
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Message sent to {topic}: partition={record_metadata.partition}, "
                        f"offset={record_metadata.offset}")
            
        except KafkaException as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            raise
    
    async def subscribe_to_topic(self, topic: str, handler: Callable,
                                group_id: str = None):
        """Subscribe to Kafka topic with message handler"""
        consumer_config = {
            'bootstrap_servers': self.config['bootstrap_servers'],
            'value_deserializer': lambda v: v.decode('utf-8'),
            'group_id': group_id or self.config.get('group_id'),
            **self.config.get('consumer_config', {})
        }
        
        consumer = KafkaConsumer(topic, **consumer_config)
        self.consumers[topic] = consumer
        self.message_handlers[topic] = handler
        
        # Start consuming in background
        asyncio.create_task(self._consume_messages(topic, consumer, handler))
        
        logger.info(f"Subscribed to topic: {topic}")
    
    async def _consume_messages(self, topic: str, consumer: KafkaConsumer,
                               handler: Callable):
        """Background task to consume messages"""
        while self.running:
            try:
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await handler(message.value)
                
            except Exception as e:
                logger.error(f"Error consuming from {topic}: {e}")
                await asyncio.sleep(1)
    
    async def health_check(self) -> bool:
        """Check Kafka connection health"""
        try:
            if self.producer:
                # Try to get metadata
                metadata = self.producer.bootstrap_connected()
                return metadata
            return False
        except Exception:
            return False
```

### Message Handlers

```python
# common/communication/message_handlers.py
import logging
from typing import Dict, Any, Callable
from ..serialization.message_serializer import MessageSerializer

logger = logging.getLogger(__name__)

class MessageHandler:
    """Base class for handling Kafka messages"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    async def handle_message(self, raw_message: str):
        """Handle incoming message"""
        try:
            message = MessageSerializer.deserialize_message(raw_message)
            message_type = message.get('type')
            
            if message_type in self.handlers:
                await self.handlers[message_type](message)
            else:
                logger.warning(f"No handler registered for message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")

class ClientMessageHandler(MessageHandler):
    """Message handler for FL clients"""
    
    def __init__(self, client):
        super().__init__()
        self.client = client
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler('global_update', self._handle_global_update)
        self.register_handler('training_start', self._handle_training_start)
        self.register_handler('evaluation_request', self._handle_evaluation_request)
        self.register_handler('shutdown', self._handle_shutdown)
    
    async def _handle_global_update(self, message: Dict[str, Any]):
        """Handle global model update from server"""
        payload = message['payload']
        round_id = payload['round_id']
        weights = payload['weights']
        
        logger.info(f"Received global update for round {round_id}")
        await self.client.process_global_update(round_id, weights)
    
    async def _handle_training_start(self, message: Dict[str, Any]):
        """Handle training start command"""
        payload = message['payload']
        logger.info("Received training start command")
        await self.client.start_training_round(payload)
    
    async def _handle_evaluation_request(self, message: Dict[str, Any]):
        """Handle evaluation request"""
        payload = message['payload']
        logger.info("Received evaluation request")
        await self.client.perform_evaluation(payload)
    
    async def _handle_shutdown(self, message: Dict[str, Any]):
        """Handle shutdown command"""
        logger.info("Received shutdown command")
        await self.client.shutdown()

class ServerMessageHandler(MessageHandler):
    """Message handler for FL server"""
    
    def __init__(self, server):
        super().__init__()
        self.server = server
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler('client_update', self._handle_client_update)
        self.register_handler('client_registration', self._handle_client_registration)
        self.register_handler('client_heartbeat', self._handle_client_heartbeat)
    
    async def _handle_client_update(self, message: Dict[str, Any]):
        """Handle client update message"""
        payload = message['payload']
        client_id = payload['client_id']
        round_id = payload['round_id']
        weights = payload['weights']
        metrics = payload['metrics']
        
        logger.info(f"Received update from client {client_id} for round {round_id}")
        await self.server.process_client_update(client_id, round_id, weights, metrics)
    
    async def _handle_client_registration(self, message: Dict[str, Any]):
        """Handle client registration"""
        payload = message['payload']
        client_id = payload['client_id']
        
        logger.info(f"Client {client_id} requesting registration")
        await self.server.register_client(client_id, payload)
    
    async def _handle_client_heartbeat(self, message: Dict[str, Any]):
        """Handle client heartbeat"""
        payload = message['payload']
        client_id = payload['client_id']
        
        await self.server.update_client_heartbeat(client_id)
```

## Utilities (`common/utils/`)

### Logging System

```python
# common/utils/logging.py
import logging
import logging.handlers
import sys
from datetime import datetime
import os

class FLLogger:
    """Federated Learning logging system"""
    
    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: str = 'INFO'):
        """Setup logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def setup_federated_logging(component_type: str, component_id: str,
                               log_dir: str = './logs'):
        """Setup logging for federated learning components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{component_type}_{component_id}_{timestamp}.log")
        
        return FLLogger.setup_logger(
            f"FL_{component_type}_{component_id}",
            log_file
        )
```

### Configuration Management

```python
# common/utils/config.py
import json
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    """Configuration management for FL system"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2)
            elif config_path.endswith(('.yml', '.yaml')):
                yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged
    
    @staticmethod
    def get_default_config(component_type: str) -> Dict[str, Any]:
        """Get default configuration for component type"""
        defaults = {
            'client': {
                'kafka': {
                    'bootstrap_servers': 'localhost:9092',
                    'group_id': 'fl_clients'
                },
                'training': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 5
                }
            },
            'server': {
                'kafka': {
                    'bootstrap_servers': 'localhost:9092'
                },
                'aggregation': {
                    'min_clients': 2,
                    'max_wait_time': 300
                }
            }
        }
        
        return defaults.get(component_type, {})
```

### Metrics Collection

```python
# common/utils/metrics.py
import time
import psutil
import torch
from typing import Dict, Any, List
from collections import defaultdict
import statistics

class MetricsCollector:
    """Collects and manages system and training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: Any, timestamp: float = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent,
            'gpu_memory': self._get_gpu_memory() if torch.cuda.is_available() else None
        }
    
    def _get_gpu_memory(self) -> Dict[str, Any]:
        """Get GPU memory information"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'total': torch.cuda.get_device_properties(0).total_memory
            }
        return None
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics"""
        return {
            'collection_duration': time.time() - self.start_time,
            'metrics': dict(self.metrics),
            'summaries': {name: self.get_metric_summary(name) 
                         for name in self.metrics.keys()}
        }
```

## Custom Exceptions (`common/exceptions/`)

```python
# common/exceptions/custom_exceptions.py

class FLException(Exception):
    """Base exception for federated learning system"""
    pass

class ModelException(FLException):
    """Exception related to model operations"""
    pass

class SerializationException(FLException):
    """Exception related to serialization/deserialization"""
    pass

class CommunicationException(FLException):
    """Exception related to Kafka communication"""
    pass

class ConfigurationException(FLException):
    """Exception related to configuration issues"""
    pass

class TrainingException(FLException):
    """Exception related to model training"""
    pass

class AggregationException(FLException):
    """Exception related to weight aggregation"""
    pass
```

## Usage Examples

### Creating a Custom Model

```python
from common.models.base import BaseFLModel
from common.models.utils import create_model_from_config

# Create model from configuration
config = {
    'type': 'SimpleMLP',
    'input_dim': 784,
    'hidden_dims': [256, 128],
    'output_dim': 10
}

model = create_model_from_config(config)
```

### Using Serialization

```python
from common.serialization.weight_serializer import WeightSerializer

# Serialize model weights
weights = model.state_dict()
serialized = WeightSerializer.serialize_weights(weights)

# Deserialize weights
restored_weights = WeightSerializer.deserialize_weights(serialized)
model.load_state_dict(restored_weights)
```

### Setting up Communication

```python
from common.communication.kafka_client import AsyncKafkaClient

# Setup Kafka client
kafka_config = {
    'bootstrap_servers': 'localhost:9092',
    'group_id': 'fl_group'
}

client = AsyncKafkaClient(kafka_config)
await client.start()

# Subscribe to topic with handler
await client.subscribe_to_topic('updates', handle_message)
```

These common modules provide the foundational components for building robust federated learning systems with proper abstraction, error handling, and extensibility.
