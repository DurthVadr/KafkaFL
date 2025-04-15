# Federated Learning with Kafka

This implementation demonstrates how to use Apache Kafka as a communication layer for federated learning with Flower.

## Prerequisites

- Python 3.8+
- Docker (for running Kafka)
- Required Python packages: `flwr`, `tensorflow`, `kafka-python`, `numpy`

## Setup

1. Install the required Python packages:
   ```
   pip install flwr tensorflow kafka-python numpy
   ```

2. Start Kafka using the provided script:
   ```
   ./scripts/start_kafka.sh
   ```
   This will start a Kafka broker on `localhost:9094` and create the required topics.

## Running the Federated Learning System

### Start the Server

```bash
python -m flwr_implementation.server --broker localhost:9094
```

Options:
- `--broker`: Kafka broker address (default: localhost:9094)
- `--grpc`: Use gRPC instead of Kafka (for comparison)
- `--num-rounds`: Number of federated learning rounds (default: 3)
- `--min-clients`: Minimum number of clients for training (default: 2)
- `--min-eval-clients`: Minimum number of clients for evaluation (default: 2)
- `--min-available-clients`: Minimum number of available clients (default: 2)
- `--fraction-fit`: Fraction of clients to sample for training (default: 0.3)
- `--fraction-eval`: Fraction of clients to sample for evaluation (default: 0.2)

### Start Clients

Start multiple clients in separate terminals:

```bash
python -m flwr_implementation.client --broker localhost:9094 --client-id 1
python -m flwr_implementation.client --broker localhost:9094 --client-id 2
# Add more clients as needed
```

Options:
- `--broker`: Kafka broker address (default: localhost:9094)
- `--client-id`: Client ID (will be randomly generated if not provided)
- `--grpc`: Use gRPC instead of Kafka (for comparison)

## Troubleshooting

If you encounter connection issues:

1. Make sure Kafka is running:
   ```
   docker ps | grep kafka-federated
   ```

2. Check if the required topics exist:
   ```
   docker exec kafka-federated kafka-topics.sh --list --bootstrap-server localhost:9092
   ```

3. Check the logs for more information:
   ```
   docker logs kafka-federated
   ```

4. If you're still having issues, try restarting Kafka:
   ```
   docker stop kafka-federated && docker rm kafka-federated
   ./scripts/start_kafka.sh
   ```

## Architecture

- `flwr_implementation/kafka_server.py`: Implements the Flower server using Kafka for communication
- `flwr_implementation/kafka_client.py`: Implements the Flower client using Kafka for communication
- `flwr_implementation/server.py`: Entry point for the server
- `flwr_implementation/client.py`: Entry point for the client
- `flwr_implementation/model.py`: Defines the machine learning model
- `flwr_implementation/dataset.py`: Handles loading and partitioning the dataset

## Common Issues

### NumPyClientWrapper Error

If you encounter an error about `NumPyClientWrapper` not existing, this is because it's an internal implementation detail in Flower. The code has been updated to use the proper `to_client()` method instead.

### Kafka Connection Errors

If you see errors like `KafkaConnectionError: 61 ECONNREFUSED`, it means the client or server cannot connect to Kafka. Make sure Kafka is running and accessible at the specified address.

### NoneType Error

If you see an error like `'NoneType' object has no attribute 'poll'`, it means the Kafka connection failed but the code tried to use it anyway. The code has been updated to handle this case properly.
