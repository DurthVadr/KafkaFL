# Local Development

## Overview

This guide covers setting up and running the federated learning system in a local development environment. It includes instructions for development setup, testing, debugging, and contributing to the project.

## Development Environment Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Java**: 8 or higher (for Kafka)
- **Git**: Latest version
- **IDE**: VS Code, PyCharm, or similar
- **Operating System**: Linux, macOS, or Windows with WSL2

### System Dependencies

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Java
brew install openjdk@11

# Install Python (if needed)
brew install python@3.9

# Install Git (if needed)
brew install git
```

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Java
sudo apt install openjdk-11-jdk

# Install Python and pip
sudo apt install python3.9 python3.9-venv python3-pip

# Install Git
sudo apt install git

# Install build tools
sudo apt install build-essential
```

#### Windows (WSL2)
```bash
# Install Windows Subsystem for Linux 2
# Follow Microsoft's official WSL2 installation guide

# After WSL2 setup, follow Ubuntu instructions above
```

### Python Environment Setup

#### Option 1: Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd /Users/mertcansaglam/cs402

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n federated-learning python=3.9

# Activate environment
conda activate federated-learning

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Development Dependencies

Create a `requirements-dev.txt` file for development-specific packages:

```txt
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0

# Development tools
ipython>=8.0.0
jupyter>=1.0.0
pre-commit>=3.0.0

# Debugging
pdb++>=0.10.3
ipdb>=0.13.0

# Performance profiling
line-profiler>=4.0.0
memory-profiler>=0.60.0
```

## Local Kafka Setup

### Option 1: Using Docker (Recommended for Development)

```bash
# Start Kafka and Zookeeper using Docker Compose
docker-compose up -d zookeeper kafka

# Verify Kafka is running
docker-compose ps

# Check Kafka logs
docker-compose logs kafka
```

### Option 2: Native Kafka Installation

#### Download and Setup Kafka

```bash
# Download Kafka
wget https://downloads.apache.org/kafka/2.13-3.5.0/kafka_2.13-3.5.0.tgz

# Extract
tar -xzf kafka_2.13-3.5.0.tgz
cd kafka_2.13-3.5.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# In another terminal, start Kafka
bin/kafka-server-start.sh config/server.properties
```

#### Using the Local Kafka Script

The project includes a convenient script for running Kafka locally:

```bash
# Make script executable
chmod +x run_local_kafka_no_docker.py

# Run local Kafka setup
python run_local_kafka_no_docker.py
```

## Development Workflow

### 1. Code Organization

```
cs402/
├── server.py              # Main server implementation
├── client.py              # Main client implementation
├── common/                 # Shared modules
│   ├── __init__.py
│   ├── data.py            # Data handling utilities
│   ├── kafka_utils.py     # Kafka communication
│   ├── logger.py          # Logging configuration
│   ├── model.py           # Model definitions
│   ├── serialization.py   # Serialization utilities
│   └── visualization.py   # Plotting and visualization
├── flower_implementation/  # Flower framework integration
├── tests/                 # Test suite
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── configs/               # Configuration files
```

### 2. Running the System Locally

#### Terminal 1: Start Kafka (if using native installation)
```bash
# Start Zookeeper
cd kafka_2.13-3.5.0
bin/zookeeper-server-start.sh config/zookeeper.properties
```

#### Terminal 2: Start Kafka Server (if using native installation)
```bash
# Start Kafka
cd kafka_2.13-3.5.0
bin/kafka-server-start.sh config/server.properties
```

#### Terminal 3: Start FL Server
```bash
# Activate virtual environment
source venv/bin/activate

# Start server
python server.py --kafka-servers localhost:9092 --min-clients 2
```

#### Terminal 4: Start FL Client 1
```bash
# Activate virtual environment
source venv/bin/activate

# Start first client
python client.py --client-id client_1 --kafka-servers localhost:9092
```

#### Terminal 5: Start FL Client 2
```bash
# Activate virtual environment
source venv/bin/activate

# Start second client
python client.py --client-id client_2 --kafka-servers localhost:9092
```

### 3. Development Scripts

Create useful development scripts in `scripts/` directory:

#### `scripts/start_dev.sh`
```bash
#!/bin/bash
# Development startup script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
else
    source venv/bin/activate
fi

# Start Kafka using Docker
echo "Starting Kafka..."
docker-compose up -d zookeeper kafka

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 10

echo "Development environment ready!"
echo "You can now start the server and clients:"
echo "  python server.py --kafka-servers localhost:9092"
echo "  python client.py --client-id client_1 --kafka-servers localhost:9092"
```

#### `scripts/test_dev.sh`
```bash
#!/bin/bash
# Development testing script

source venv/bin/activate

# Run code quality checks
echo "Running code quality checks..."
black --check .
flake8 .
isort --check-only .
mypy .

# Run tests
echo "Running tests..."
pytest tests/ -v --cov=. --cov-report=html

echo "All checks completed!"
```

#### `scripts/clean_dev.sh`
```bash
#!/bin/bash
# Clean development environment

# Stop Docker containers
docker-compose down

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean logs
rm -rf logs/*.log

# Clean plots
rm -rf plots/*.png

echo "Development environment cleaned!"
```

## IDE Configuration

### VS Code Setup

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.pytest_cache": true
    }
}
```

Create `.vscode/launch.json` for debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FL Server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/server.py",
            "args": ["--kafka-servers", "localhost:9092", "--min-clients", "2"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "FL Client",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/client.py",
            "args": ["--client-id", "debug_client", "--kafka-servers", "localhost:9092"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### PyCharm Setup

1. Open the project in PyCharm
2. Configure Python interpreter to use `./venv/bin/python`
3. Install plugins: Black, Pylint, Mypy
4. Configure code style to use Black formatter
5. Set up run configurations for server and client

## Testing and Quality Assurance

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_server.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

### Code Quality Tools

#### Black (Code Formatting)
```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .

# Format specific file
black server.py
```

#### Flake8 (Linting)
```bash
# Lint all Python files
flake8 .

# Lint specific file
flake8 server.py

# Lint with specific configuration
flake8 --max-line-length=88 --extend-ignore=E203,W503 .
```

#### isort (Import Sorting)
```bash
# Sort imports in all files
isort .

# Check import sorting
isort --check-only .

# Sort imports with Black compatibility
isort --profile black .
```

#### MyPy (Type Checking)
```bash
# Type check all files
mypy .

# Type check specific file
mypy server.py

# Type check with strict settings
mypy --strict .
```

### Pre-commit Hooks

Set up pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Debugging

### Using Python Debugger

#### With pdb
```python
import pdb

def train_model(self):
    pdb.set_trace()  # Debugger will stop here
    # Your code here
```

#### With ipdb (Enhanced debugger)
```python
import ipdb

def train_model(self):
    ipdb.set_trace()  # Enhanced debugger
    # Your code here
```

### Logging for Development

#### Enhanced Development Logging
```python
import logging
from common.logger import setup_logger

# Setup development logger
logger = setup_logger(
    name="dev_logger",
    log_file="logs/development.log",
    level="DEBUG"
)

# Use throughout development
logger.debug("Detailed debugging information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

#### Kafka Message Debugging
```python
# Enable Kafka debug logging
import logging
logging.getLogger('kafka').setLevel(logging.DEBUG)

# Log all Kafka messages
def debug_kafka_message(message):
    logger.debug(f"Kafka message: {message}")
    return message
```

### Performance Profiling

#### Line Profiler
```bash
# Install line profiler
pip install line_profiler

# Profile specific function
@profile
def train_model(self):
    # Your code here
    pass

# Run with profiler
kernprof -l -v server.py
```

#### Memory Profiler
```bash
# Install memory profiler
pip install memory_profiler

# Profile memory usage
@profile
def train_model(self):
    # Your code here
    pass

# Run with memory profiler
python -m memory_profiler server.py
```

## Data Management for Development

### Sample Data Generation

Create development datasets:

```python
# scripts/generate_dev_data.py
import numpy as np
import torch
from sklearn.datasets import make_classification
import os

def generate_client_data(client_id, num_samples=1000, num_features=20, num_classes=10):
    """Generate synthetic data for development"""
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_classes=num_classes,
        n_clusters_per_class=1,
        random_state=client_id  # Different data for each client
    )
    
    # Create client data directory
    client_dir = f"data/client_{client_id}"
    os.makedirs(client_dir, exist_ok=True)
    
    # Save data
    np.save(f"{client_dir}/X_train.npy", X)
    np.save(f"{client_dir}/y_train.npy", y)
    
    print(f"Generated data for client_{client_id}: {X.shape}, {y.shape}")

# Generate data for multiple clients
for i in range(1, 6):
    generate_client_data(i)
```

### Data Validation

```python
# scripts/validate_data.py
import os
import numpy as np

def validate_client_data():
    """Validate client data integrity"""
    
    data_dir = "data"
    clients = []
    
    for item in os.listdir(data_dir):
        if item.startswith("client_"):
            clients.append(item)
    
    print(f"Found {len(clients)} client directories")
    
    for client in clients:
        client_path = os.path.join(data_dir, client)
        
        # Check required files
        required_files = ["X_train.npy", "y_train.npy"]
        for file in required_files:
            file_path = os.path.join(client_path, file)
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
        
        # Load and validate data
        try:
            X = np.load(os.path.join(client_path, "X_train.npy"))
            y = np.load(os.path.join(client_path, "y_train.npy"))
            
            print(f"{client}: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
            
        except Exception as e:
            print(f"Error loading {client} data: {e}")

if __name__ == "__main__":
    validate_client_data()
```

## Contributing Guidelines

### Development Process

1. **Fork and Clone**
   ```bash
   git clone <your-fork-url>
   cd cs402
   git remote add upstream <original-repo-url>
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

4. **Test Changes**
   ```bash
   # Run full test suite
   ./scripts/test_dev.sh
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### Code Style Guidelines

- Use Black for code formatting
- Follow PEP 8 guidelines
- Use type hints where possible
- Write descriptive docstrings
- Keep functions small and focused
- Use meaningful variable names

### Documentation Standards

- Update documentation for new features
- Include code examples
- Write clear commit messages
- Update changelog for significant changes

## Troubleshooting Development Issues

### Common Problems

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="${PYTHONPATH}:/Users/mertcansaglam/cs402"
   
   # Or add to your shell profile
   echo 'export PYTHONPATH="${PYTHONPATH}:/Users/mertcansaglam/cs402"' >> ~/.zshrc
   ```

2. **Kafka Connection Issues**
   ```bash
   # Check if Kafka is running
   docker-compose ps
   
   # Restart Kafka
   docker-compose restart kafka
   
   # Check Kafka logs
   docker-compose logs kafka
   ```

3. **Port Conflicts**
   ```bash
   # Check what's using port 9092
   lsof -i :9092
   
   # Kill process if necessary
   kill -9 <PID>
   ```

4. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Performance Issues

- Use profiling tools to identify bottlenecks
- Monitor memory usage during development
- Optimize data loading and processing
- Use appropriate batch sizes for training

This comprehensive local development guide provides everything needed to set up, develop, test, and contribute to the federated learning project efficiently.
