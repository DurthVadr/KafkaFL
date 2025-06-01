# Installation Guide

This guide covers detailed installation instructions for different environments and deployment scenarios.

## System Requirements

### Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores, 10GB disk space
- **Recommended**: 8GB RAM, 4 CPU cores, 20GB disk space
- **For multiple clients**: Additional 1GB RAM per client

### Software Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12 (Python 3.13 not supported)
- **Docker**: 20.10+ (optional, for containerized deployment)
- **Docker Compose**: 2.0+ (optional)

## Installation Methods

### Method 1: Python Virtual Environment (Recommended)

#### 1. Create Virtual Environment

```bash
# Using venv
python -m venv federated_learning_env
source federated_learning_env/bin/activate  # Linux/macOS
# federated_learning_env\Scripts\activate   # Windows

# Using conda
conda create -n federated_learning python=3.11
conda activate federated_learning
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Verify Installation

```bash
python -c "import tensorflow; import kafka; print('All dependencies installed successfully')"
```

### Method 2: Docker Environment

#### 1. Install Docker

##### Ubuntu/Debian
```bash
sudo apt update
sudo apt install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
```

##### macOS
```bash
# Install Docker Desktop from https://docker.com/products/docker-desktop
# Or using Homebrew
brew install --cask docker
```

##### Windows
Download and install Docker Desktop from https://docker.com/products/docker-desktop

#### 2. Verify Docker Installation

```bash
docker --version
docker compose version
```

#### 3. Build Project Images

```bash
docker compose build
```

### Method 3: System-Wide Installation

```bash
# Install Python dependencies system-wide (not recommended for production)
pip install -r requirements.txt

# On Ubuntu/Debian, you might need:
sudo apt install python3-dev python3-pip
```

## Dependency Details

### Core Dependencies

```txt
tensorflow==2.15.0          # Deep learning framework
numpy==1.24.3               # Numerical computing
kafka-python==2.0.2         # Kafka client library
scikit-learn==1.3.2         # Machine learning utilities
scipy==1.11.4               # Scientific computing
matplotlib==3.8.2           # Plotting library
seaborn==0.13.0              # Statistical visualization
flwr==1.5.0                 # Flower framework (for comparison)
```

### Optional Dependencies

For development and testing:
```bash
pip install pytest pytest-cov black flake8 mypy
```

## Environment Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Kafka Configuration
BOOTSTRAP_SERVERS=localhost:9094
KAFKA_LOG_LEVEL=WARN

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=-1  # Force CPU-only

# Application Configuration
REDUCED_DATA_SIZE=0
CLIENT_ID=auto
DURATION_MINUTES=60
AGGREGATION_INTERVAL_SECONDS=60
TRAINING_INTERVAL_SECONDS=120
MIN_UPDATES_PER_AGGREGATION=1
```

### Kafka Setup

#### Local Kafka Installation

##### Using Docker (Recommended)
```bash
# Start Kafka using the provided docker-compose
docker compose up kafka zookeeper -d
```

##### Manual Installation
```bash
# Download Kafka
wget https://downloads.apache.org/kafka/2.13-3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties &

# Start Kafka
bin/kafka-server-start.sh config/server.properties &
```

## Platform-Specific Instructions

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install TensorFlow dependencies
pip install tensorflow-macos tensorflow-metal  # For Apple Silicon Macs
```

### Windows with WSL2

```bash
# Install WSL2 with Ubuntu
wsl --install

# Inside WSL2
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Ubuntu/Debian

```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev build-essential

# Install system dependencies for TensorFlow
sudo apt install libhdf5-dev libc-ares-dev libeigen3-dev
```

### CentOS/RHEL/Fedora

```bash
# Install Python and development tools
sudo dnf install python3.11 python3.11-devel gcc gcc-c++ make

# Install TensorFlow dependencies
sudo dnf install hdf5-devel
```

## Verification and Testing

### 1. Basic Functionality Test

```bash
# Test server startup
python server.py --help

# Test client startup
python client.py --help

# Test local runner
python run_local_kafka_no_docker.py --help
```

### 2. Component Tests

```bash
# Test Kafka connectivity
python -c "
from common.kafka_utils import create_producer, create_consumer
producer = create_producer('localhost:9094')
print('Kafka connection successful' if producer else 'Kafka connection failed')
"

# Test TensorFlow
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
"

# Test model creation
python -c "
from common.model import create_lenet_model
model = create_lenet_model()
print(f'Model created with {model.count_params()} parameters')
"
```

### 3. Integration Test

```bash
# Run a short test
python run_local_kafka_no_docker.py --duration 2 --num-clients 2
```

## Troubleshooting Installation Issues

### Common Issues

#### TensorFlow Installation Issues

```bash
# For older CPUs without AVX support
pip install tensorflow-cpu==2.15.0

# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal
```

#### Kafka Connection Issues

```bash
# Check if Kafka is running
docker ps | grep kafka

# Check Kafka logs
docker logs <kafka_container_id>

# Test Kafka manually
docker exec -it <kafka_container_id> kafka-topics.sh --list --bootstrap-server localhost:9092
```

#### Memory Issues

```bash
# Reduce TensorFlow memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Use reduced dataset
export REDUCED_DATA_SIZE=1
```

#### Port Conflicts

```bash
# Find processes using ports
lsof -i :9092  # Kafka
lsof -i :9094  # Alternative Kafka port
lsof -i :2181  # Zookeeper

# Kill processes if needed
sudo kill -9 <PID>
```

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review log files in the `logs/` directory
3. Open an issue on the project repository
4. Check system compatibility in the requirements section

## Next Steps

After successful installation:

1. Follow the [Quick Start Guide](quick_start.md)
2. Read the [Configuration Guide](configuration.md)
3. Explore the [Federated Learning Process](federated_learning_process.md)
