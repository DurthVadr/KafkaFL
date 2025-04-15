#!/bin/bash

# This script helps start a Kafka server for testing the federated learning system

echo "Starting Kafka for Federated Learning..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Start Kafka using Docker
echo "Starting Kafka using Docker..."
docker run -d \
    --name kafka-federated \
    -p 9094:9094 \
    -e KAFKA_CFG_NODE_ID=0 \
    -e KAFKA_CFG_PROCESS_ROLES=controller,broker \
    -e KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093,EXTERNAL://:9094 \
    -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka-federated:9092,EXTERNAL://localhost:9094 \
    -e KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,EXTERNAL:PLAINTEXT \
    -e KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka-federated:9093 \
    -e KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER \
    bitnami/kafka:latest

# Wait for Kafka to start
echo "Waiting for Kafka to start..."
sleep 10

# Create topics
echo "Creating topics..."
docker exec kafka-federated kafka-topics.sh --create --topic model_topic --bootstrap-server localhost:9092
docker exec kafka-federated kafka-topics.sh --create --topic update_topic --bootstrap-server localhost:9092

echo "Kafka is ready for federated learning!"
echo "To stop Kafka, run: docker stop kafka-federated && docker rm kafka-federated"
