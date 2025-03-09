# Federated Learning with Kafka

This project implements a federated learning system using Kafka as the communication layer between clients and a server. The server manages a global model, while clients train the model on their local datasets and send updates back to the server.

## Project Structure

- `server.py`: Implements the `FederatedServer` class, which manages the global model and coordinates updates from clients.
- `client.py`: Implements the `FederatedClient` class, which trains a local model on the CIFAR-10 dataset and communicates with the server.
- `docker-compose.yml`: (Optional) Defines the Docker configuration for running Kafka and Kafka UI.

## Requirements

- Python 3.x -3.13 not supported-
- Kafka
- TensorFlow
- scikit-learn
- NumPy

## Setup Instructions

### 1. Install Dependencies
Make sure you have the required Python packages installed. You can use pip to install them:

```bash
pip install numpy tensorflow scikit-learn kafka-python
```

### 2. Run Kafka
You can run Kafka using Docker with the provided `docker-compose.yml` file. Navigate to the directory containing the file and run:

```bash
docker-compose up
```

### 3. Start the Server
In one terminal, run the server:

```bash
python server.py
```

### 4. Start the Client
In another terminal, run the client:

```bash
python client.py
```

You can run multiple instances of the client to simulate multiple clients in the federated learning setup.

## Usage

1. The server initializes a global model and listens for updates from clients.
2. Each client registers with the server, receives the initial model, trains on local data, and sends the updated model back to the server.
3. The server aggregates the updates and sends the new global model back to the clients.


# To Do List

1. Fix the `WARNING:kafka.conn:DNS lookup failed for 7c96e78b7cd5:9092, exception was [Errno 8] nodename nor servname provided, or not known. Is your advertised.listeners (called advertised.host.name before Kafka 9) correct and resolvable?` error and succesfully implement synchronous federated learning
2. Get the synch fl benchmarks and compare with Flower 
3. kafka-pyhon library is very weak in async tasks. It is not like the official java library. Check for aiokafka and if async is not possible with current architecture change the code than library.
4. Check for async methods like Heartbeat or buffer. There are papers
5. Succesfully implement Async module
6. create tests and benchmarks

