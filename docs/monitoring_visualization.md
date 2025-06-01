# Monitoring and Visualization

## Overview

This document covers the comprehensive monitoring and visualization capabilities of the federated learning system. The system provides real-time monitoring of training progress, system metrics, and performance analytics through various visualization tools and dashboards.

## Monitoring Architecture

### Components Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FL Clients    │    │   FL Server     │    │   Kafka Broker  │
│                 │    │                 │    │                 │
│ • Local Metrics │    │ • Global Metrics│    │ • Message Stats │
│ • Performance   │    │ • Aggregation   │    │ • Throughput    │
│ • Resource Use  │    │ • Round Stats   │    │ • Lag Metrics   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │   Monitoring System     │
                    │                         │
                    │ • Metrics Collection    │
                    │ • Data Aggregation      │
                    │ • Visualization         │
                    │ • Alerting              │
                    └─────────────────────────┘
```

### Metrics Categories

1. **Training Metrics**: Loss, accuracy, convergence rates
2. **System Metrics**: CPU, memory, GPU usage
3. **Communication Metrics**: Message throughput, latency
4. **Business Metrics**: Client participation, round completion times

## Built-in Visualization System

### Visualization Module (`common/visualization.py`)

The system includes a comprehensive visualization module for creating plots and dashboards:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class FLVisualizer:
    """Federated Learning visualization system"""
    
    def __init__(self, output_dir: str = './plots'):
        self.output_dir = output_dir
        self.setup_style()
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Custom color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd'
        }
    
    def plot_training_progress(self, metrics_data: Dict[str, List[float]], 
                             save_path: str = None):
        """Plot training progress over rounds"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Federated Learning Training Progress', fontsize=16)
        
        # Global loss
        if 'global_loss' in metrics_data:
            axes[0, 0].plot(metrics_data['global_loss'], 
                          color=self.colors['primary'], linewidth=2)
            axes[0, 0].set_title('Global Model Loss')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Global accuracy
        if 'global_accuracy' in metrics_data:
            axes[0, 1].plot(metrics_data['global_accuracy'], 
                          color=self.colors['success'], linewidth=2)
            axes[0, 1].set_title('Global Model Accuracy')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Client participation
        if 'client_participation' in metrics_data:
            axes[1, 0].bar(range(len(metrics_data['client_participation'])), 
                          metrics_data['client_participation'],
                          color=self.colors['info'], alpha=0.7)
            axes[1, 0].set_title('Client Participation per Round')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Number of Clients')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Round duration
        if 'round_duration' in metrics_data:
            axes[1, 1].plot(metrics_data['round_duration'], 
                          color=self.colors['warning'], linewidth=2)
            axes[1, 1].set_title('Round Duration')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Duration (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{self.output_dir}/training_progress.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_client_metrics(self, client_data: Dict[str, Dict[str, List[float]]]):
        """Plot per-client metrics"""
        
        num_clients = len(client_data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Client-Specific Metrics', fontsize=16)
        
        # Client losses
        axes[0, 0].set_title('Client Local Losses')
        for client_id, metrics in client_data.items():
            if 'loss' in metrics:
                axes[0, 0].plot(metrics['loss'], label=f'Client {client_id}', 
                              linewidth=1.5, alpha=0.8)
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Client accuracies
        axes[0, 1].set_title('Client Local Accuracies')
        for client_id, metrics in client_data.items():
            if 'accuracy' in metrics:
                axes[0, 1].plot(metrics['accuracy'], label=f'Client {client_id}',
                              linewidth=1.5, alpha=0.8)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training times
        axes[1, 0].set_title('Client Training Times')
        for client_id, metrics in client_data.items():
            if 'training_time' in metrics:
                axes[1, 0].plot(metrics['training_time'], label=f'Client {client_id}',
                              linewidth=1.5, alpha=0.8)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Data samples per client
        sample_counts = [len(metrics.get('loss', [])) for metrics in client_data.values()]
        client_ids = list(client_data.keys())
        
        axes[1, 1].bar(client_ids, sample_counts, color=self.colors['secondary'], alpha=0.7)
        axes[1, 1].set_title('Training Rounds per Client')
        axes[1, 1].set_xlabel('Client ID')
        axes[1, 1].set_ylabel('Number of Rounds')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/client_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_system_metrics(self, system_data: Dict[str, List[float]]):
        """Plot system performance metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Metrics', fontsize=16)
        
        # CPU usage
        if 'cpu_usage' in system_data:
            axes[0, 0].plot(system_data['cpu_usage'], color=self.colors['primary'])
            axes[0, 0].set_title('CPU Usage Over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('CPU Usage (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Memory usage
        if 'memory_usage' in system_data:
            axes[0, 1].plot(system_data['memory_usage'], color=self.colors['warning'])
            axes[0, 1].set_title('Memory Usage Over Time')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Memory Usage (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Network throughput
        if 'network_throughput' in system_data:
            axes[1, 0].plot(system_data['network_throughput'], color=self.colors['success'])
            axes[1, 0].set_title('Network Throughput')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Throughput (MB/s)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # GPU usage (if available)
        if 'gpu_usage' in system_data:
            axes[1, 1].plot(system_data['gpu_usage'], color=self.colors['info'])
            axes[1, 1].set_title('GPU Usage Over Time')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('GPU Usage (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/system_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, metrics_data: Dict[str, Any]):
        """Create interactive Plotly dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Global Loss', 'Global Accuracy', 'Client Participation',
                          'Round Duration', 'System Metrics', 'Communication Stats'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Global loss
        if 'global_loss' in metrics_data:
            fig.add_trace(
                go.Scatter(y=metrics_data['global_loss'], name='Global Loss',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Global accuracy
        if 'global_accuracy' in metrics_data:
            fig.add_trace(
                go.Scatter(y=metrics_data['global_accuracy'], name='Global Accuracy',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
        
        # Client participation
        if 'client_participation' in metrics_data:
            fig.add_trace(
                go.Bar(y=metrics_data['client_participation'], name='Client Participation',
                       marker_color='orange'),
                row=2, col=1
            )
        
        # Round duration
        if 'round_duration' in metrics_data:
            fig.add_trace(
                go.Scatter(y=metrics_data['round_duration'], name='Round Duration',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
        
        # System metrics
        if 'cpu_usage' in metrics_data:
            fig.add_trace(
                go.Scatter(y=metrics_data['cpu_usage'], name='CPU Usage',
                          line=dict(color='purple', width=2)),
                row=3, col=1
            )
        
        if 'memory_usage' in metrics_data:
            fig.add_trace(
                go.Scatter(y=metrics_data['memory_usage'], name='Memory Usage',
                          line=dict(color='brown', width=2)),
                row=3, col=1, secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title='Federated Learning Interactive Dashboard',
            height=900,
            showlegend=True
        )
        
        # Save interactive plot
        fig.write_html(f"{self.output_dir}/interactive_dashboard.html")
        
        return fig
    
    def plot_convergence_analysis(self, convergence_data: Dict[str, List[float]]):
        """Plot convergence analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Convergence Analysis', fontsize=16)
        
        # Loss convergence
        if 'loss_values' in convergence_data:
            axes[0, 0].semilogy(convergence_data['loss_values'])
            axes[0, 0].set_title('Loss Convergence (Log Scale)')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Loss (log scale)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norms
        if 'gradient_norms' in convergence_data:
            axes[0, 1].plot(convergence_data['gradient_norms'])
            axes[0, 1].set_title('Gradient Norms')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Weight changes
        if 'weight_changes' in convergence_data:
            axes[1, 0].plot(convergence_data['weight_changes'])
            axes[1, 0].set_title('Weight Changes Between Rounds')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('L2 Norm of Weight Change')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if 'learning_rates' in convergence_data:
            axes[1, 1].plot(convergence_data['learning_rates'])
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
```

### Real-time Monitoring

```python
class RealTimeMonitor:
    """Real-time monitoring system for federated learning"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.metrics_buffer = {}
        self.visualizer = FLVisualizer()
        self.running = False
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        
        while self.running:
            # Collect current metrics
            current_metrics = await self.collect_current_metrics()
            
            # Update metrics buffer
            self.update_metrics_buffer(current_metrics)
            
            # Generate updated visualizations
            self.generate_real_time_plots()
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def collect_current_metrics(self):
        """Collect current system and training metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            metrics['gpu_usage'] = torch.cuda.utilization()
            metrics['gpu_memory'] = torch.cuda.memory_usage()
        
        return metrics
    
    def update_metrics_buffer(self, new_metrics: Dict[str, Any]):
        """Update metrics buffer with new data"""
        for key, value in new_metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            
            self.metrics_buffer[key].append(value)
            
            # Keep only last 100 data points
            if len(self.metrics_buffer[key]) > 100:
                self.metrics_buffer[key] = self.metrics_buffer[key][-100:]
    
    def generate_real_time_plots(self):
        """Generate real-time plot updates"""
        if len(self.metrics_buffer) > 0:
            self.visualizer.plot_system_metrics(self.metrics_buffer)
```

## Kafka Monitoring

### Message Flow Visualization

```python
class KafkaMonitor:
    """Monitor Kafka message flow and performance"""
    
    def __init__(self, kafka_config: Dict[str, str]):
        self.kafka_config = kafka_config
        self.message_stats = defaultdict(list)
        self.throughput_stats = defaultdict(list)
    
    def monitor_topic_throughput(self, topic: str):
        """Monitor message throughput for a topic"""
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda v: v.decode('utf-8')
        )
        
        message_count = 0
        start_time = time.time()
        
        for message in consumer:
            message_count += 1
            current_time = time.time()
            
            # Calculate throughput every 10 seconds
            if current_time - start_time >= 10:
                throughput = message_count / (current_time - start_time)
                self.throughput_stats[topic].append({
                    'timestamp': current_time,
                    'throughput': throughput,
                    'message_count': message_count
                })
                
                message_count = 0
                start_time = current_time
    
    def plot_kafka_metrics(self):
        """Plot Kafka performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Kafka Performance Metrics', fontsize=16)
        
        # Message throughput
        for topic, stats in self.throughput_stats.items():
            timestamps = [s['timestamp'] for s in stats]
            throughput = [s['throughput'] for s in stats]
            
            axes[0, 0].plot(timestamps, throughput, label=f'{topic}')
        
        axes[0, 0].set_title('Message Throughput by Topic')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Messages/Second')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Message latency (if available)
        # Consumer lag metrics
        # Partition metrics
        
        plt.tight_layout()
        plt.savefig('plots/kafka_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
```

## External Monitoring Integration

### Prometheus Integration

```python
# monitoring/prometheus_exporter.py
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import time

class PrometheusExporter:
    """Export federated learning metrics to Prometheus"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Define metrics
        self.global_loss = Gauge('fl_global_loss', 'Global model loss')
        self.global_accuracy = Gauge('fl_global_accuracy', 'Global model accuracy')
        self.client_count = Gauge('fl_active_clients', 'Number of active clients')
        self.round_duration = Histogram('fl_round_duration_seconds', 'FL round duration')
        self.messages_sent = Counter('fl_messages_sent_total', 'Total messages sent')
        self.cpu_usage = Gauge('fl_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('fl_memory_usage_percent', 'Memory usage percentage')
    
    def start_server(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        print(f"Prometheus metrics server started on port {self.port}")
    
    def update_training_metrics(self, loss: float, accuracy: float, 
                              client_count: int, round_duration: float):
        """Update training metrics"""
        self.global_loss.set(loss)
        self.global_accuracy.set(accuracy)
        self.client_count.set(client_count)
        self.round_duration.observe(round_duration)
    
    def update_system_metrics(self, cpu_percent: float, memory_percent: float):
        """Update system metrics"""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_percent)
    
    def increment_message_counter(self):
        """Increment message counter"""
        self.messages_sent.inc()
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "Federated Learning Dashboard",
    "tags": ["federated-learning", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Global Model Loss",
        "type": "stat",
        "targets": [
          {
            "expr": "fl_global_loss",
            "legendFormat": "Global Loss"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 1.0}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Global Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "fl_global_accuracy",
            "legendFormat": "Global Accuracy"
          }
        ]
      },
      {
        "id": 3,
        "title": "Active Clients",
        "type": "stat",
        "targets": [
          {
            "expr": "fl_active_clients",
            "legendFormat": "Active Clients"
          }
        ]
      },
      {
        "id": 4,
        "title": "Training Progress",
        "type": "graph",
        "targets": [
          {
            "expr": "fl_global_loss",
            "legendFormat": "Loss"
          },
          {
            "expr": "fl_global_accuracy",
            "legendFormat": "Accuracy"
          }
        ],
        "yAxes": [
          {
            "label": "Loss",
            "min": 0
          },
          {
            "label": "Accuracy (%)",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "id": 5,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "fl_cpu_usage_percent",
            "legendFormat": "CPU Usage (%)"
          },
          {
            "expr": "fl_memory_usage_percent",
            "legendFormat": "Memory Usage (%)"
          }
        ]
      },
      {
        "id": 6,
        "title": "Round Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fl_round_duration_seconds_sum[5m]) / rate(fl_round_duration_seconds_count[5m])",
            "legendFormat": "Average Round Duration"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

## Alerting System

### Alert Configuration

```python
class AlertManager:
    """Manage alerts for federated learning system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
        self.notification_handlers = []
    
    def add_alert_rule(self, name: str, condition: callable, 
                      severity: str, message: str):
        """Add alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'last_triggered': None
        })
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert conditions"""
        for rule in self.alert_rules:
            if rule['condition'](metrics):
                if self._should_trigger_alert(rule):
                    self._trigger_alert(rule, metrics)
    
    def _should_trigger_alert(self, rule: Dict[str, Any]) -> bool:
        """Check if alert should be triggered (debouncing)"""
        if rule['last_triggered'] is None:
            return True
        
        # Don't trigger same alert within 5 minutes
        return time.time() - rule['last_triggered'] > 300
    
    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger alert"""
        alert_data = {
            'name': rule['name'],
            'severity': rule['severity'],
            'message': rule['message'],
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        # Send to all notification handlers
        for handler in self.notification_handlers:
            handler.send_alert(alert_data)
        
        rule['last_triggered'] = time.time()

# Example alert rules
def setup_default_alerts(alert_manager: AlertManager):
    """Setup default alert rules"""
    
    # High loss alert
    alert_manager.add_alert_rule(
        name="high_global_loss",
        condition=lambda m: m.get('global_loss', 0) > 2.0,
        severity="warning",
        message="Global model loss is unusually high"
    )
    
    # Low client participation
    alert_manager.add_alert_rule(
        name="low_client_participation",
        condition=lambda m: m.get('active_clients', 0) < 2,
        severity="critical",
        message="Too few clients participating in training"
    )
    
    # High system resource usage
    alert_manager.add_alert_rule(
        name="high_cpu_usage",
        condition=lambda m: m.get('cpu_usage', 0) > 90,
        severity="warning",
        message="CPU usage is critically high"
    )
    
    # Training stall detection
    alert_manager.add_alert_rule(
        name="training_stalled",
        condition=lambda m: m.get('rounds_without_improvement', 0) > 10,
        severity="critical",
        message="Training appears to have stalled"
    )
```

## Performance Analytics

### Training Performance Analysis

```python
class PerformanceAnalyzer:
    """Analyze federated learning performance"""
    
    def __init__(self):
        self.metrics_history = []
        self.analysis_results = {}
    
    def analyze_convergence(self, loss_history: List[float]) -> Dict[str, Any]:
        """Analyze convergence characteristics"""
        
        if len(loss_history) < 5:
            return {'status': 'insufficient_data'}
        
        # Calculate convergence rate
        recent_losses = loss_history[-5:]
        improvement_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
        
        # Detect convergence
        variance = np.var(recent_losses)
        is_converged = variance < 0.001 and improvement_rate < 0.01
        
        # Estimate rounds to convergence
        if not is_converged and improvement_rate > 0:
            current_loss = recent_losses[-1]
            target_loss = 0.1  # Configurable
            rounds_to_target = (current_loss - target_loss) / improvement_rate
        else:
            rounds_to_target = None
        
        return {
            'is_converged': is_converged,
            'convergence_rate': improvement_rate,
            'variance': variance,
            'rounds_to_target': rounds_to_target,
            'current_loss': recent_losses[-1]
        }
    
    def analyze_client_performance(self, client_metrics: Dict[str, Dict[str, List[float]]]):
        """Analyze per-client performance"""
        
        analysis = {}
        
        for client_id, metrics in client_metrics.items():
            if 'training_time' in metrics:
                avg_training_time = np.mean(metrics['training_time'])
                training_time_std = np.std(metrics['training_time'])
                
                analysis[client_id] = {
                    'avg_training_time': avg_training_time,
                    'training_time_consistency': 1 / (1 + training_time_std),
                    'participation_rate': len(metrics['training_time']) / len(client_metrics),
                    'performance_trend': self._calculate_trend(metrics['training_time'])
                }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in performance values"""
        if len(values) < 3:
            return 'unknown'
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'deteriorating'
        elif slope < -0.1:
            return 'improving'
        else:
            return 'stable'
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        report = """
# Federated Learning Performance Report

## Executive Summary
- Training Status: {status}
- Current Global Loss: {global_loss:.4f}
- Current Global Accuracy: {global_accuracy:.2f}%
- Active Clients: {active_clients}

## Convergence Analysis
- Convergence Status: {convergence_status}
- Convergence Rate: {convergence_rate:.6f}
- Estimated Rounds to Target: {rounds_to_target}

## Client Performance
- Average Training Time: {avg_training_time:.2f}s
- Most Consistent Client: {best_client}
- Slowest Client: {slowest_client}

## Recommendations
{recommendations}
        """.format(
            status=self.analysis_results.get('status', 'Unknown'),
            global_loss=self.analysis_results.get('global_loss', 0),
            global_accuracy=self.analysis_results.get('global_accuracy', 0),
            active_clients=self.analysis_results.get('active_clients', 0),
            convergence_status=self.analysis_results.get('convergence_status', 'Unknown'),
            convergence_rate=self.analysis_results.get('convergence_rate', 0),
            rounds_to_target=self.analysis_results.get('rounds_to_target', 'Unknown'),
            avg_training_time=self.analysis_results.get('avg_training_time', 0),
            best_client=self.analysis_results.get('best_client', 'Unknown'),
            slowest_client=self.analysis_results.get('slowest_client', 'Unknown'),
            recommendations=self._generate_recommendations()
        )
        
        return report
    
    def _generate_recommendations(self) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        # Add specific recommendations based on analysis
        if self.analysis_results.get('convergence_rate', 0) < 0.001:
            recommendations.append("- Consider increasing learning rate or adjusting model architecture")
        
        if self.analysis_results.get('avg_training_time', 0) > 60:
            recommendations.append("- Training time is high, consider reducing batch size or model complexity")
        
        if self.analysis_results.get('active_clients', 0) < 3:
            recommendations.append("- Low client participation, investigate client connectivity issues")
        
        return '\n'.join(recommendations) if recommendations else "- System performance is optimal"
```

## Usage Examples

### Basic Monitoring Setup

```python
import asyncio
from monitoring.visualizer import FLVisualizer
from monitoring.real_time_monitor import RealTimeMonitor

async def main():
    # Setup visualizer
    visualizer = FLVisualizer(output_dir='./monitoring_output')
    
    # Setup real-time monitor
    monitor = RealTimeMonitor(update_interval=10)
    
    # Start monitoring
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    # Let it run for demonstration
    await asyncio.sleep(60)
    
    # Stop monitoring
    monitor.running = False
    await monitoring_task

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with FL Server

```python
# In server.py
from monitoring.prometheus_exporter import PrometheusExporter
from monitoring.visualizer import FLVisualizer

class FederatedLearningServer:
    def __init__(self):
        # ...existing code...
        
        # Setup monitoring
        self.prometheus_exporter = PrometheusExporter(port=8000)
        self.visualizer = FLVisualizer()
        self.prometheus_exporter.start_server()
    
    async def complete_round(self, round_id: int):
        # ...existing training code...
        
        # Update monitoring metrics
        self.prometheus_exporter.update_training_metrics(
            loss=self.global_loss,
            accuracy=self.global_accuracy,
            client_count=len(self.active_clients),
            round_duration=round_duration
        )
        
        # Generate plots
        metrics_data = {
            'global_loss': self.loss_history,
            'global_accuracy': self.accuracy_history,
            'client_participation': self.participation_history
        }
        
        self.visualizer.plot_training_progress(metrics_data)
```

This comprehensive monitoring and visualization system provides deep insights into federated learning performance, enabling effective optimization and troubleshooting of the training process.
