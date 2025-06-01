# Flower Integration

## Overview

This document covers the integration of the Flower framework with the federated learning system. Flower (Federated Learning Flower) is a friendly federated learning framework that provides additional capabilities for managing federated learning workflows, client strategies, and advanced aggregation methods.

## Flower Framework Architecture

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Federated Learning System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Kafka-based   │    │     Flower      │    │   Monitoring    │ │
│  │  Communication  │◄──►│   Framework     │◄──►│  & Metrics      │ │
│  │                 │    │                 │    │                 │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│           ▼                       ▼                       ▼         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ FL Server       │    │ Flower Server   │    │ Visualization   │ │
│  │ (Kafka-based)   │    │ (Strategy Mgmt) │    │ Dashboard       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                                 │
│           ▼                       ▼                                 │
│  ┌─────────────────┐    ┌─────────────────┐                       │
│  │ FL Clients      │    │ Flower Clients  │                       │
│  │ (Kafka-based)   │    │ (gRPC-based)    │                       │
│  └─────────────────┘    └─────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Flower Implementation Structure

The Flower integration is located in the `flower_implementation/` directory:

```
flower_implementation/
├── __init__.py
├── client_app.py      # Flower client application
├── server_app.py      # Flower server application
├── task.py           # Task definitions and utilities
└── strategies/       # Custom aggregation strategies
    ├── __init__.py
    ├── fedavg_kafka.py
    └── adaptive_strategies.py
```

## Flower Client Implementation

### Client Application (`flower_implementation/client_app.py`)

```python
import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import OrderedDict

from common.model import create_model
from common.data import load_client_data
from flower_implementation.task import get_model_parameters, set_model_parameters

class FlowerFLClient(fl.client.NumPyClient):
    """Flower client implementation for federated learning"""
    
    def __init__(self, client_id: str, model_config: Dict, data_config: Dict):
        self.client_id = client_id
        self.model_config = model_config
        self.data_config = data_config
        
        # Initialize model
        self.model = create_model(model_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load client data
        self.train_loader, self.test_loader = self.load_data()
        
        # Training configuration
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=model_config.get('learning_rate', 0.001)
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.local_epochs = model_config.get('local_epochs', 5)
        self.training_metrics = []
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare client data"""
        return load_client_data(
            client_id=self.client_id,
            data_path=self.data_config['data_path'],
            batch_size=self.data_config.get('batch_size', 32),
            validation_split=self.data_config.get('validation_split', 0.2)
        )
    
    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Return current model parameters as NumPy arrays"""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update model parameters from NumPy arrays"""
        set_model_parameters(self.model, parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model locally and return updated parameters"""
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Extract training configuration from server
        epochs = int(config.get("local_epochs", self.local_epochs))
        learning_rate = float(config.get("learning_rate", 0.001))
        
        # Update learning rate if changed
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Train the model
        train_loss, train_accuracy = self.train_model(epochs)
        
        # Collect training metrics
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "client_id": self.client_id,
            "local_epochs": epochs,
            "samples": len(self.train_loader.dataset)
        }
        
        # Return updated parameters and metrics
        return (
            self.get_parameters({}),
            len(self.train_loader.dataset),
            metrics
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data"""
        
        # Set parameters for evaluation
        self.set_parameters(parameters)
        
        # Evaluate model
        test_loss, test_accuracy = self.evaluate_model()
        
        # Return loss, number of samples, and metrics
        return (
            test_loss,
            len(self.test_loader.dataset),
            {"accuracy": test_accuracy, "client_id": self.client_id}
        )
    
    def train_model(self, epochs: int) -> Tuple[float, float]:
        """Train model locally for specified epochs"""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
            
            # Log epoch progress
            epoch_accuracy = 100.0 * epoch_correct / epoch_total
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            print(f"Client {self.client_id} - Epoch {epoch + 1}/{epochs}: "
                  f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        avg_loss = total_loss / (epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_model(self) -> Tuple[float, float]:
        """Evaluate model on test data"""
        
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy

def create_flower_client(client_id: str) -> FlowerFLClient:
    """Factory function to create Flower client"""
    
    # Load configuration
    model_config = {
        'model_type': 'SimpleMLP',
        'input_dim': 784,
        'hidden_dims': [128, 64],
        'output_dim': 10,
        'learning_rate': 0.001,
        'local_epochs': 5
    }
    
    data_config = {
        'data_path': f'./data/client_{client_id}',
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    return FlowerFLClient(client_id, model_config, data_config)
```

## Flower Server Implementation

### Server Application (`flower_implementation/server_app.py`)

```python
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import Parameters
import torch
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import json

from common.model import create_model
from flower_implementation.task import get_model_parameters
from flower_implementation.strategies.fedavg_kafka import FedAvgKafka

class FlowerFLServer:
    """Flower server implementation with Kafka integration"""
    
    def __init__(self, server_config: Dict):
        self.server_config = server_config
        self.model_config = server_config['model']
        
        # Initialize global model
        self.global_model = create_model(self.model_config)
        
        # Server configuration
        self.min_clients = server_config.get('min_clients', 2)
        self.min_available_clients = server_config.get('min_available_clients', 2)
        self.num_rounds = server_config.get('num_rounds', 10)
        
        # Metrics storage
        self.round_metrics = []
        self.global_metrics = {
            'losses': [],
            'accuracies': [],
            'participation': []
        }
        
        # Setup strategy
        self.strategy = self.create_strategy()
    
    def create_strategy(self) -> fl.server.strategy.Strategy:
        """Create federated learning strategy"""
        
        # Get initial global parameters
        initial_parameters = get_model_parameters(self.global_model)
        
        # Create strategy with Kafka integration
        strategy = FedAvgKafka(
            fraction_fit=self.server_config.get('fraction_fit', 1.0),
            fraction_evaluate=self.server_config.get('fraction_evaluate', 1.0),
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_available_clients,
            min_available_clients=self.min_available_clients,
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            evaluate_fn=self.get_evaluate_fn(),
            on_fit_config_fn=self.get_fit_config_fn(),
            on_evaluate_config_fn=self.get_evaluate_config_fn(),
            kafka_config=self.server_config.get('kafka', {})
        )
        
        return strategy
    
    def get_evaluate_fn(self):
        """Return evaluation function for server-side evaluation"""
        
        def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, fl.common.Scalar]):
            """Evaluate global model on server"""
            
            # Convert parameters to model weights
            params_list = fl.common.parameters_to_ndarrays(parameters)
            
            # Set global model parameters
            from flower_implementation.task import set_model_parameters
            set_model_parameters(self.global_model, params_list)
            
            # Evaluate on server test data (if available)
            loss, accuracy = self.evaluate_global_model()
            
            # Store metrics
            self.global_metrics['losses'].append(loss)
            self.global_metrics['accuracies'].append(accuracy)
            
            print(f"Round {server_round} - Global Loss: {loss:.4f}, Global Accuracy: {accuracy:.2f}%")
            
            return loss, {"accuracy": accuracy, "round": server_round}
        
        return evaluate_fn
    
    def get_fit_config_fn(self):
        """Return function that configures each round of training"""
        
        def fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
            """Configure training for each round"""
            
            config = {
                "server_round": server_round,
                "local_epochs": self.model_config.get('local_epochs', 5),
                "learning_rate": self.model_config.get('learning_rate', 0.001)
            }
            
            # Adaptive learning rate
            if server_round > 5:
                config["learning_rate"] *= 0.95  # Decay learning rate
            
            return config
        
        return fit_config_fn
    
    def get_evaluate_config_fn(self):
        """Return function that configures each round of evaluation"""
        
        def evaluate_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
            """Configure evaluation for each round"""
            return {"server_round": server_round}
        
        return evaluate_config_fn
    
    def evaluate_global_model(self) -> Tuple[float, float]:
        """Evaluate global model on server test data"""
        
        # Load server test data (implement based on your setup)
        try:
            from common.data import load_server_test_data
            test_loader = load_server_test_data()
            
            self.global_model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.global_model.to(device)
            
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.global_model(data)
                    
                    test_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            avg_loss = test_loss / len(test_loader)
            accuracy = 100.0 * correct / total
            
            return avg_loss, accuracy
            
        except Exception as e:
            print(f"Server evaluation not available: {e}")
            return 0.0, 0.0
    
    def start_server(self):
        """Start Flower federated learning server"""
        
        print(f"Starting Flower FL Server with {self.min_clients} minimum clients")
        print(f"Model: {self.model_config}")
        
        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy
        )
    
    def save_results(self, output_path: str = "./flower_results"):
        """Save training results and metrics"""
        
        Path(output_path).mkdir(exist_ok=True)
        
        # Save global metrics
        metrics_file = Path(output_path) / "global_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.global_metrics, f, indent=2)
        
        # Save model
        model_file = Path(output_path) / "global_model.pth"
        torch.save(self.global_model.state_dict(), model_file)
        
        # Generate visualizations
        self.generate_visualizations(output_path)
        
        print(f"Results saved to {output_path}")
    
    def generate_visualizations(self, output_path: str):
        """Generate training visualizations"""
        
        import matplotlib.pyplot as plt
        
        if not self.global_metrics['losses']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.global_metrics['losses'], 'b-', linewidth=2)
        ax1.set_title('Global Model Loss')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.global_metrics['accuracies'], 'g-', linewidth=2)
        ax2.set_title('Global Model Accuracy')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_path) / "training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_flower_server(config_path: str = None) -> FlowerFLServer:
    """Factory function to create Flower server"""
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            server_config = json.load(f)
    else:
        # Default configuration
        server_config = {
            'model': {
                'model_type': 'SimpleMLP',
                'input_dim': 784,
                'hidden_dims': [128, 64],
                'output_dim': 10,
                'learning_rate': 0.001,
                'local_epochs': 5
            },
            'min_clients': 2,
            'min_available_clients': 2,
            'num_rounds': 10,
            'fraction_fit': 1.0,
            'fraction_evaluate': 1.0,
            'kafka': {
                'bootstrap_servers': 'localhost:9092',
                'topic_prefix': 'flower_fl'
            }
        }
    
    return FlowerFLServer(server_config)
```

## Custom Aggregation Strategies

### Kafka-Integrated FedAvg Strategy

```python
# flower_implementation/strategies/fedavg_kafka.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, FitRes, EvaluateRes
from typing import Dict, List, Tuple, Optional, Union
import asyncio
from kafka import KafkaProducer, KafkaConsumer
import json
import threading

class FedAvgKafka(FedAvg):
    """FedAvg strategy with Kafka integration for metrics and coordination"""
    
    def __init__(self, kafka_config: Dict, **kwargs):
        super().__init__(**kwargs)
        
        self.kafka_config = kafka_config
        self.topic_prefix = kafka_config.get('topic_prefix', 'flower_fl')
        
        # Initialize Kafka producer for metrics
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Metrics storage
        self.round_metrics = []
        self.aggregation_metrics = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate training results with Kafka metrics reporting"""
        
        # Extract client metrics
        client_metrics = []
        for client, fit_res in results:
            if fit_res.metrics:
                client_metrics.append({
                    'client_id': fit_res.metrics.get('client_id', 'unknown'),
                    'train_loss': fit_res.metrics.get('train_loss', 0),
                    'train_accuracy': fit_res.metrics.get('train_accuracy', 0),
                    'samples': fit_res.metrics.get('samples', 0),
                    'local_epochs': fit_res.metrics.get('local_epochs', 0)
                })
        
        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Calculate round statistics
        round_stats = {
            'round': server_round,
            'participating_clients': len(results),
            'failed_clients': len(failures),
            'total_samples': sum(fit_res.num_examples for _, fit_res in results),
            'avg_train_loss': sum(m['train_loss'] for m in client_metrics) / len(client_metrics) if client_metrics else 0,
            'avg_train_accuracy': sum(m['train_accuracy'] for m in client_metrics) / len(client_metrics) if client_metrics else 0,
            'client_metrics': client_metrics
        }
        
        # Store metrics
        self.round_metrics.append(round_stats)
        
        # Send metrics to Kafka
        self._send_metrics_to_kafka('round_complete', round_stats)
        
        print(f"Round {server_round} aggregation complete - "
              f"Clients: {len(results)}, Avg Loss: {round_stats['avg_train_loss']:.4f}")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        """Aggregate evaluation results with Kafka metrics reporting"""
        
        # Extract evaluation metrics
        eval_metrics = []
        for client, evaluate_res in results:
            if evaluate_res.metrics:
                eval_metrics.append({
                    'client_id': evaluate_res.metrics.get('client_id', 'unknown'),
                    'test_loss': evaluate_res.loss,
                    'test_accuracy': evaluate_res.metrics.get('accuracy', 0),
                    'samples': evaluate_res.num_examples
                })
        
        # Perform standard evaluation aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Calculate evaluation statistics
        eval_stats = {
            'round': server_round,
            'global_test_loss': aggregated_loss,
            'participating_clients': len(results),
            'total_test_samples': sum(evaluate_res.num_examples for _, evaluate_res in results),
            'client_eval_metrics': eval_metrics
        }
        
        # Send evaluation metrics to Kafka
        self._send_metrics_to_kafka('evaluation_complete', eval_stats)
        
        return aggregated_loss, aggregated_metrics
    
    def _send_metrics_to_kafka(self, event_type: str, metrics: Dict):
        """Send metrics to Kafka topic"""
        
        try:
            topic = f"{self.topic_prefix}_{event_type}"
            message = {
                'event_type': event_type,
                'timestamp': import time; time.time(),
                'metrics': metrics
            }
            
            self.producer.send(topic, value=message)
            self.producer.flush()
            
        except Exception as e:
            print(f"Failed to send metrics to Kafka: {e}")
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure training round with adaptive parameters"""
        
        # Get base configuration
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)
        
        # Add adaptive configuration based on previous rounds
        if len(self.round_metrics) > 0:
            last_round = self.round_metrics[-1]
            avg_loss = last_round['avg_train_loss']
            
            # Adaptive learning rate based on loss trend
            if len(self.round_metrics) >= 2:
                prev_loss = self.round_metrics[-2]['avg_train_loss']
                if avg_loss > prev_loss:  # Loss increased
                    # Reduce learning rate
                    for client, fit_ins in fit_configurations:
                        if fit_ins.config:
                            current_lr = fit_ins.config.get('learning_rate', 0.001)
                            fit_ins.config['learning_rate'] = current_lr * 0.9
        
        return fit_configurations
```

## Task Utilities

### Model Parameter Handling (`flower_implementation/task.py`)

```python
import torch
import torch.nn as nn
from typing import List
import numpy as np
from collections import OrderedDict

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from numpy arrays"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def calculate_model_size(model: nn.Module) -> Dict[str, int]:
    """Calculate model size information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory usage (rough estimate)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'memory_bytes': param_bytes + buffer_bytes,
        'memory_mb': (param_bytes + buffer_bytes) / (1024 * 1024)
    }

def compare_model_parameters(params1: List[np.ndarray], 
                           params2: List[np.ndarray]) -> Dict[str, float]:
    """Compare two sets of model parameters"""
    
    if len(params1) != len(params2):
        raise ValueError("Parameter lists must have same length")
    
    differences = []
    l2_norms = []
    
    for p1, p2 in zip(params1, params2):
        diff = p1 - p2
        differences.append(diff)
        l2_norms.append(np.linalg.norm(diff))
    
    total_diff_norm = np.sqrt(sum(norm**2 for norm in l2_norms))
    max_diff_norm = max(l2_norms)
    avg_diff_norm = np.mean(l2_norms)
    
    return {
        'total_l2_norm': total_diff_norm,
        'max_layer_norm': max_diff_norm,
        'avg_layer_norm': avg_diff_norm,
        'num_layers': len(params1)
    }
```

## Running Flower Integration

### Server Startup Script

```python
# flower_runner.py
import argparse
import json
from pathlib import Path
from flower_implementation.server_app import create_flower_server

def main():
    parser = argparse.ArgumentParser(description="Run Flower FL Server")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients")
    parser.add_argument("--output", type=str, default="./flower_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and start server
    server = create_flower_server(args.config)
    
    # Override configuration with command line arguments
    if args.rounds:
        server.num_rounds = args.rounds
    if args.min_clients:
        server.min_clients = args.min_clients
    
    try:
        # Start server
        server.start_server()
        
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        # Save results
        server.save_results(args.output)

if __name__ == "__main__":
    main()
```

### Client Startup Script

```python
# flower_client_runner.py
import argparse
import flwr as fl
from flower_implementation.client_app import create_flower_client

def main():
    parser = argparse.ArgumentParser(description="Run Flower FL Client")
    parser.add_argument("--client-id", type=str, required=True, 
                       help="Client ID")
    parser.add_argument("--server-address", type=str, 
                       default="localhost:8080", 
                       help="Server address")
    parser.add_argument("--data-path", type=str, 
                       help="Path to client data")
    
    args = parser.parse_args()
    
    # Create client
    client = create_flower_client(args.client_id)
    
    # Start client
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()
```

## Hybrid Integration with Kafka

### Coordinated Execution

```python
class HybridFLCoordinator:
    """Coordinate between Kafka-based and Flower-based FL systems"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kafka_server = None
        self.flower_server = None
        
    async def start_hybrid_training(self):
        """Start coordinated FL training with both systems"""
        
        # Start Kafka-based system for communication
        kafka_task = asyncio.create_task(self.start_kafka_system())
        
        # Start Flower system for advanced strategies
        flower_task = asyncio.create_task(self.start_flower_system())
        
        # Wait for both systems
        await asyncio.gather(kafka_task, flower_task)
    
    async def start_kafka_system(self):
        """Start Kafka-based FL system"""
        from server import FederatedLearningServer
        
        self.kafka_server = FederatedLearningServer(self.config['kafka'])
        await self.kafka_server.start()
    
    async def start_flower_system(self):
        """Start Flower-based FL system"""
        from flower_implementation.server_app import create_flower_server
        
        self.flower_server = create_flower_server(self.config['flower'])
        # Run in separate process to avoid blocking
        import multiprocessing
        p = multiprocessing.Process(target=self.flower_server.start_server)
        p.start()
        p.join()
```

## Metrics Integration

### Unified Metrics Collection

```python
class UnifiedMetricsCollector:
    """Collect metrics from both Kafka and Flower systems"""
    
    def __init__(self):
        self.kafka_metrics = []
        self.flower_metrics = []
        self.kafka_consumer = None
    
    def start_kafka_metrics_collection(self):
        """Start collecting metrics from Kafka topics"""
        
        self.kafka_consumer = KafkaConsumer(
            'flower_fl_round_complete',
            'flower_fl_evaluation_complete',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        
        for message in self.kafka_consumer:
            self.flower_metrics.append(message.value)
    
    def export_unified_metrics(self, output_path: str):
        """Export unified metrics from both systems"""
        
        unified_metrics = {
            'kafka_metrics': self.kafka_metrics,
            'flower_metrics': self.flower_metrics,
            'combined_analysis': self.analyze_combined_metrics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(unified_metrics, f, indent=2)
    
    def analyze_combined_metrics(self) -> Dict:
        """Analyze metrics from both systems"""
        
        return {
            'total_rounds': len(self.flower_metrics),
            'system_comparison': self.compare_system_performance(),
            'convergence_analysis': self.analyze_convergence()
        }
```

## Usage Examples

### Running Flower Server

```bash
# Start Flower server
python flower_runner.py --rounds 20 --min-clients 3 --output ./flower_results

# Start with custom configuration
python flower_runner.py --config configs/flower_server.json
```

### Running Flower Clients

```bash
# Start multiple Flower clients
python flower_client_runner.py --client-id client_1 &
python flower_client_runner.py --client-id client_2 &
python flower_client_runner.py --client-id client_3 &
```

### Hybrid System

```python
# Start hybrid FL system
config = {
    'kafka': {
        'bootstrap_servers': 'localhost:9092',
        'min_clients': 2
    },
    'flower': {
        'num_rounds': 10,
        'min_clients': 2
    }
}

coordinator = HybridFLCoordinator(config)
await coordinator.start_hybrid_training()
```

This comprehensive Flower integration provides advanced federated learning capabilities while maintaining compatibility with the existing Kafka-based communication system, enabling sophisticated aggregation strategies and enhanced monitoring capabilities.
