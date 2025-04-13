# Federated Learning with Flower and Kafka

This implementation provides a federated learning solution using the Flower (flwr) framework with Kafka as the communication backend.

## Overview

This implementation is based on the FDxKafka approach and includes:

- A server component that coordinates the federated learning process
- A client component that trains models on local data
- Support for both Kafka and gRPC communication
- A simple CIFAR-10 image classification task as an example

## Requirements

Install the required dependencies:

```bash
pip install -r flwr_implementation/requirements.txt
```

## Usage

### Running with the Script

The easiest way to run the federated learning system is using the provided script:

```bash
./flwr_implementation/run_federated_learning.sh --broker localhost:9094 --num-clients 3 --num-rounds 5
```

Options:
- `--broker`: Kafka broker address (default: localhost:9094)
- `--num-clients`: Number of clients to start (default: 3)
- `--num-rounds`: Number of federated learning rounds (default: 3)
- `--grpc`: Use gRPC instead of Kafka
- `--min-clients`: Minimum number of clients required (default: 2)

### Running Manually

#### Start the Server

```bash
python flwr_implementation/server.py --broker localhost:9094 --num-rounds 5
```

For gRPC:
```bash
python flwr_implementation/server.py --broker localhost:8080 --grpc --num-rounds 5
```

#### Start Clients

```bash
python flwr_implementation/client.py --broker localhost:9094 --client-id 1
python flwr_implementation/client.py --broker localhost:9094 --client-id 2
python flwr_implementation/client.py --broker localhost:9094 --client-id 3
```

For gRPC:
```bash
python flwr_implementation/client.py --broker localhost:8080 --grpc --client-id 1
```

## Architecture

- **Server**: Coordinates the federated learning process, aggregates model updates
- **Client**: Trains models on local data, sends updates to the server
- **Model**: Simple CNN for CIFAR-10 image classification
- **Dataset**: Partitioned CIFAR-10 dataset to simulate distributed data

## Customization

You can customize this implementation by:

1. Modifying the model architecture in `model.py`
2. Changing the dataset or partitioning strategy in `dataset.py`
3. Adjusting the federated learning parameters in `server.py`
4. Modifying the client training process in `client.py`
