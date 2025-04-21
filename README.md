# Federated Learning with Kafka

This project implements a federated learning system using Kafka as the communication layer between clients and a server. The server manages a global model, while clients train the model on their local datasets and send updates back to the server.

## Overview

Federated Learning is a machine learning approach where multiple clients (e.g., mobile devices, edge devices) train a shared model while keeping their data local. This implementation uses Kafka as the communication infrastructure to enable scalable and reliable message passing between the server and clients.

## Project Structure

### Core Components
- `server.py`: Implements the `FederatedServer` class, which manages the global model and coordinates updates from clients.
- `client.py`: Implements the `FederatedClient` class, which trains a local model on the CIFAR-10 dataset and communicates with the server.

### Common Modules
- `common/model.py`: Defines the CNN model architecture for CIFAR-10 classification.
- `common/lightweight_model.py`: Provides a smaller model option that uses fewer resources.
- `common/data.py`: Handles loading and preprocessing the CIFAR-10 dataset.
- `common/serialization.py`: Provides utilities for serializing and deserializing model weights.
- `common/kafka_utils.py`: Contains functions for Kafka communication.
- `common/logger.py`: Implements a custom logging system with colored output.

### Deployment
- `docker-compose.yml`: Defines the Docker configuration for running Kafka, the server, and multiple clients.
- `Dockerfile.server`: Docker configuration for the server.
- `Dockerfile.client`: Docker configuration for the clients.

### Utilities
- `run_local_kafka.py`: Alternative to Docker for running the system with less resource usage.
- `scripts/start_kafka.sh`: Helper script for starting Kafka manually.
- `test_federated_learning.sh`: Script for testing the federated learning system.

## Features

- **Federated Averaging**: Implements the FedAvg algorithm to aggregate model updates from multiple clients.
- **Weight Compatibility**: Includes mechanisms to handle and adapt weights when there are minor architecture differences between server and client models.
- **Robust Communication**: Uses Kafka for reliable message passing between server and clients.
- **Model Serialization**: Efficient serialization and deserialization of model weights for transmission over Kafka.
- **Accuracy Evaluation**: Clients evaluate model performance on test data after training.
- **Lightweight Model Option**: Provides a smaller model that uses fewer resources for constrained environments.
- **Comprehensive Logging**: Includes a custom logging system with colored output and file logging.
- **Resource Optimization**: Includes options to reduce memory usage and optimize performance.

## Requirements

- Python 3.x (Python 3.13 not supported)
- Docker and Docker Compose (for Docker-based deployment)
- Kafka
- TensorFlow
- scikit-learn
- NumPy
- kafka-python

## Setup Instructions

### 1. Install Dependencies
Make sure you have the required Python packages installed. You can use pip to install them:

```bash
pip install -r requirements.txt
```

### 2. Run with Docker (Resource-Intensive but Simple)
You can run the entire system (Kafka, server, and clients) with a single Docker Compose command:

```bash
docker compose build
docker compose up
```

This will start:
- A Kafka broker
- The federated learning server
- Multiple federated learning clients

### 3. Run Locally (Less Resource-Intensive)
For environments with limited resources, you can use the local script:

```bash
python run_local_kafka.py
```

This script:
- Checks if Kafka is running locally
- Starts the server and multiple clients with optimized resource settings
- Uses environment variables to reduce memory usage

### 4. Run Individual Components (Optional)

If you want to run components separately:

#### Start Kafka
```bash
# Using Docker
docker compose up kafka

# Or using the script
./scripts/start_kafka.sh
```

#### Start the Server
```bash
python server.py
```

#### Start a Client
```bash
python client.py
```

You can run multiple instances of the client to simulate multiple clients in the federated learning setup.

## How It Works

### Federated Learning Process

1. **Initialization**: The server initializes a global model and publishes it to the `model_topic` Kafka topic.

2. **Client Training**: Each client:
   - Subscribes to the `model_topic` to receive the global model
   - Loads the CIFAR-10 dataset (or a subset) for training
   - Adapts the global model weights if necessary to match its local architecture
   - Trains the model on its local data for a specified number of epochs
   - Evaluates the model on local test data to measure accuracy
   - Sends the updated model weights to the `update_topic`

3. **Aggregation**: The server:
   - Collects model updates from clients via the `update_topic`
   - Performs federated averaging to create an updated global model
   - Publishes the new global model back to the `model_topic`

4. **Iteration**: Steps 2-3 repeat for multiple rounds to improve the global model.

5. **Evaluation**: After multiple rounds of training, clients evaluate the final model on test data.

### Weight Adaptation Mechanism

The system includes sophisticated mechanisms to handle cases where the model architecture might slightly differ between server and clients:

1. **Weight Compatibility Check**: Verifies if the received weights can be directly applied to the local model by comparing shapes and dimensions.

2. **Weight Adaptation**: When there's a mismatch (particularly in the flattened layer dimensions), the system attempts to adapt the weights by:
   - Analyzing the layer structure and identifying compatible layers
   - Padding or truncating weights as needed to fit the local model
   - Preserving as much learned information as possible

3. **Partial Update**: When only some layers are compatible, the system can apply partial updates to those layers while keeping other layers unchanged.

4. **Fallback Mechanism**: If adaptation isn't possible, the system can fall back to using a fresh model as a last resort.

### Serialization Process

The system uses an efficient serialization process to transmit model weights over Kafka:

1. **Weight Extraction**: Extracts numpy arrays from the TensorFlow model
2. **Binary Serialization**: Converts arrays to binary format with metadata
3. **Compression**: Optionally compresses the data to reduce message size
4. **Integrity Verification**: Includes checksums to verify data integrity
5. **Deserialization**: Reconstructs the original arrays on the receiving end

## Configuration

### Environment Variables

The system supports several environment variables for configuration:

- `BOOTSTRAP_SERVERS`: Kafka bootstrap servers (default: "localhost:9094")
- `CLIENT_ID`: ID for the client (default: randomly generated)
- `USE_LIGHTWEIGHT_MODEL`: Set to "1" to use the lightweight model (default: "0")
- `REDUCED_DATA_SIZE`: Set to "1" to use a smaller dataset for faster training (default: "0")
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging level (default: "2" for warnings only)

### Docker Configuration

- **Number of Clients**: Modify the `docker-compose.yml` file to add or remove client services.
- **Memory Limits**: Adjust the memory limits in `docker-compose.yml` based on your system resources.
- **Volume Mounts**: Configure persistent storage for logs and data.

### Model and Training Configuration

- **Training Parameters**: Adjust batch size, learning rate, and other parameters in the client code.
- **Model Architecture**: Modify the model architecture in `common/model.py` or use the lightweight model option.
- **Number of Rounds**: Change the number of federated learning rounds in `server.py`.

## Troubleshooting

### Common Issues

1. **Kafka Connection Issues**:
   - Ensure Kafka is running and accessible. Check the bootstrap server address.
   - Verify that the required topics exist using `kafka-topics.sh --list --bootstrap-server localhost:9092`
   - Check Kafka logs for connection errors

2. **Weight Compatibility Errors**:
   - If you see errors about incompatible weights, check that the model architectures in server.py and client.py match exactly.
   - Try setting `USE_LIGHTWEIGHT_MODEL=1` for both server and clients to ensure consistent architecture.
   - Check the model version in `common/model.py` to ensure all components are using the same version.

3. **Memory Issues**:
   - If you encounter memory problems with Docker, try reducing the memory limits in `docker-compose.yml`
   - Use the lightweight model option by setting `USE_LIGHTWEIGHT_MODEL=1`
   - Reduce the dataset size by setting `REDUCED_DATA_SIZE=1`
   - Use `run_local_kafka.py` instead of Docker for a more resource-efficient setup

4. **TensorFlow Errors**:
   - If you see CUDA or GPU-related errors, try setting `CUDA_VISIBLE_DEVICES=-1` to force CPU-only mode
   - For version compatibility issues, check the exact TensorFlow version in `requirements.txt`

5. **Docker Issues**:
   - If Docker containers crash, check the logs with `docker logs <container_id>`
   - Increase the memory allocated to Docker in Docker Desktop settings
   - Try running components individually without Docker

## Future Work

1. ✅ Fix the server-client communication
2. Get the synchronous FL benchmarks and compare with Flower
3. Improve async capabilities - kafka-python library has limitations for async tasks compared to the Java library. Consider aiokafka if the current architecture needs async support.
4. Implement advanced async methods like Heartbeat or buffer mechanisms
5. Successfully implement an Async module
6. Create comprehensive tests and benchmarks
7. Add support for differential privacy
8. Implement secure aggregation protocols
9. Add support for more complex model architectures
10. Implement client selection strategies
11. Add support for heterogeneous client devices
12. Implement model compression techniques to reduce communication overhead

## Documentation

Detailed documentation is available in the `docs` directory:

- [Federated Learning Process](docs/federated_learning_process.md): Detailed explanation of the federated learning algorithm
- [Model Architecture](docs/model_architecture.md): Information about the CNN architecture for CIFAR-10
- [Weight Adaptation Mechanism](docs/weight_adaptation.md): Details on how weight compatibility is handled

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
