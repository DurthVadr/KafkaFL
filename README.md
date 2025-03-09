# Federated Learning with Kafka

This project implements a federated learning system using Kafka as the communication layer between clients and a server. The server manages a global model, while clients train the model on their local datasets and send updates back to the server.

## Project Structure

- `server.py`: Implements the `FederatedServer` class, which manages the global model and coordinates updates from clients.
- `client.py`: Implements the `FederatedClient` class, which trains a local model on the CIFAR-10 dataset and communicates with the server.
- `docker-compose.yml`: (Optional) Defines the Docker configuration for running Kafka and Kafka UI.

## Requirements

- Python 3.x
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
2. Each client registers with the server, receives the initial model, t
