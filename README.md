# Federated Learning with Kafka

This project implements a federated learning system using Kafka as the communication layer between clients and a server. The server manages a global model, while clients train the model on their local datasets and send updates back to the server.

## Overview

Federated Learning is a machine learning approach where multiple clients (e.g., mobile devices, edge devices) train a shared model while keeping their data local. This implementation uses Kafka as the communication infrastructure to enable scalable and reliable message passing between the server and clients.

## Project Structure

- `Server/server.py`: Implements the `FederatedServer` class, which manages the global model and coordinates updates from clients.
- `Client/client.py`: Implements the `FederatedClient` class, which trains a local model on the CIFAR-10 dataset and communicates with the server.
- `docker-compose.yml`: Defines the Docker configuration for running Kafka, Kafka UI, the server, and multiple clients.

## Features

- **Federated Averaging**: Implements the FedAvg algorithm to aggregate model updates from multiple clients.
- **Weight Compatibility**: Includes mechanisms to handle and adapt weights when there are minor architecture differences between server and client models.
- **Robust Communication**: Uses Kafka for reliable message passing between server and clients.
- **Model Serialization**: Efficient serialization and deserialization of model weights for transmission over Kafka.
- **Accuracy Evaluation**: Clients evaluate model performance on test data after training.

## Requirements

- Python 3.x (Python 3.13 not supported)
- Docker and Docker Compose
- Kafka
- TensorFlow
- scikit-learn
- NumPy
- kafka-python

## Setup Instructions

### 1. Install Dependencies
Make sure you have the required Python packages installed. You can use pip to install them:

```bash
pip install numpy tensorflow scikit-learn kafka-python
```

### 2. Run the Federated Learning System
You can run the entire system (Kafka, server, and clients) with a single Docker Compose command:

```bash
docker compose build
docker compose up
```

This will start:
- A Kafka broker
- A Kafka UI for monitoring (accessible at http://localhost:8080)
- The federated learning server
- Multiple federated learning clients

### 3. Run Individual Components (Optional)

If you want to run components separately:

#### Start Kafka
```bash
docker compose up kafka kafka-ui
```

#### Start the Server
```bash
python Server/server.py
```

#### Start a Client
```bash
python Client/client.py
```

You can run multiple instances of the client to simulate multiple clients in the federated learning setup.

## How It Works

### Federated Learning Process

1. **Initialization**: The server initializes a global model and publishes it to the `model_topic` Kafka topic.

2. **Client Training**: Each client:
   - Subscribes to the `model_topic` to receive the global model
   - Loads the CIFAR-10 dataset (or a subset) for training
   - Adapts the global model weights if necessary to match its local architecture
   - Trains the model on its local data
   - Sends the updated model weights to the `update_topic`

3. **Aggregation**: The server:
   - Collects model updates from clients via the `update_topic`
   - Performs federated averaging to create an updated global model
   - Publishes the new global model back to the `model_topic`

4. **Evaluation**: After multiple rounds of training, clients evaluate the final model on test data.

### Weight Adaptation

The system includes mechanisms to handle cases where the model architecture might slightly differ between server and clients:

1. **Weight Compatibility Check**: Verifies if the received weights can be directly applied to the local model.

2. **Weight Adaptation**: When there's a mismatch (particularly in the flattened layer dimensions), the system attempts to adapt the weights by padding or truncating as needed.

3. **Fallback Mechanism**: If adaptation isn't possible, the system can fall back to using a fresh model as a last resort.

## Configuration

You can configure various aspects of the system:

- **Number of Clients**: Modify the `docker-compose.yml` file to add or remove client services.
- **Training Parameters**: Adjust batch size, learning rate, and other parameters in the client code.
- **Model Architecture**: Modify the model architecture in both server and client code (ensure they remain compatible).

## Troubleshooting

### Common Issues

1. **Kafka Connection Issues**: Ensure Kafka is running and accessible. Check the bootstrap server address.

2. **Weight Compatibility Errors**: If you see errors about incompatible weights, check that the model architectures in server.py and client.py match exactly.

3. **Memory Issues**: If you encounter memory problems, try reducing the dataset size or model complexity.

## Future Work

1. ✅ Fix the server-client communication
2. Get the synchronous FL benchmarks and compare with Flower
3. Improve async capabilities - kafka-python library has limitations for async tasks compared to the Java library. Consider aiokafka if the current architecture needs async support.
4. Implement advanced async methods like Heartbeat or buffer mechanisms
5. Successfully implement an Async module
6. Create comprehensive tests and benchmarks
7. Add support for differential privacy
8. Implement secure aggregation protocols

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
