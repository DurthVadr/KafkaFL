# Quick Start Guide

This guide will help you get the federated learning system up and running quickly.

## Prerequisites

- Python 3.8+ (Python 3.13 not supported)
- Docker and Docker Compose (for containerized deployment)
- Git

## Option 1: Docker Deployment (Recommended for First-Time Users)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cs402
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Complete System

```bash
# Build and start all services
docker compose build
docker compose up
```

This will automatically start:
- Kafka broker
- Federated learning server
- 3 federated learning clients

### 4. Monitor Progress

Watch the console output for:
- Client training progress
- Server aggregation events
- Accuracy improvements over time

### 5. View Results

Results and visualizations will be saved in the `plots/` directory.

## Option 2: Local Development (Resource-Efficient)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Local System

```bash
# Run with default settings
python run_local_kafka_no_docker.py

# Or customize parameters
python run_local_kafka_no_docker.py --duration 10 --aggregation-interval 20 --training-interval 40 --num-clients 5
```

### Parameters Explained

- `--duration`: How long to run (minutes)
- `--aggregation-interval`: Time between server aggregations (seconds)
- `--training-interval`: Time between client training cycles (seconds)
- `--num-clients`: Number of client instances to start

## What to Expect

### Console Output

You'll see logs showing:
```
[INFO] Server: Starting federated learning server
[INFO] Client 1: Training local model
[INFO] Server: Aggregating 3 model updates
[INFO] Client 2: Model evaluation: accuracy=0.4567
```

### Generated Files

- `logs/`: Detailed log files for each component
- `plots/`: Performance visualizations and metrics
- `data/`: CIFAR-10 dataset (automatically downloaded)

### Typical Performance

- Initial accuracy: ~10% (random)
- After 5-10 rounds: ~40-60%
- Training time per round: 30-60 seconds per client

## Next Steps

1. **Explore Configurations**: See [Configuration Guide](configuration.md)
2. **Understand the Process**: Read [Federated Learning Process](federated_learning_process.md)
3. **Compare with Flower**: Try [Flower Integration](flower_integration.md)
4. **Advanced Features**: Learn about [Asynchronous FL](asynchronous_fl.md)

## Troubleshooting Quick Fixes

### Kafka Connection Issues
```bash
# Check if Kafka is running
docker ps | grep kafka

# Restart Kafka if needed
docker compose restart kafka
```

### Memory Issues
```bash
# Use reduced dataset
export REDUCED_DATA_SIZE=1
python run_local_kafka_no_docker.py
```

### Port Conflicts
```bash
# Check port usage
lsof -i :9092  # Kafka
lsof -i :9094  # Alternative Kafka port
```

For detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md).
