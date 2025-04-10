#!/bin/bash

# Run Federated Learning with 3 clients

# Make sure Kafka is running
echo "Starting Kafka and server using docker-compose..."
cd Server
docker-compose build
docker-compose up -d

# Wait for Kafka and server to start
echo "Waiting for Kafka and server to start..."
sleep 60  # Increased wait time to ensure Kafka is fully started

# Check if Kafka is running
echo "Checking if Kafka is accessible..."
for i in {1..5}; do
  if nc -z localhost 9094; then
    echo "Kafka is accessible on port 9094"
    break
  else
    echo "Waiting for Kafka to be accessible on port 9094 (attempt $i/5)..."
    sleep 10
  fi
done

# Run 3 client instances
echo "Starting 3 client instances..."
cd ../Client

# Run client 1
export CLIENT_ID=1
export BOOTSTRAP_SERVERS=localhost:9094
python client.py &
CLIENT1_PID=$!

# Run client 2
export CLIENT_ID=2
export BOOTSTRAP_SERVERS=localhost:9094
python client.py &
CLIENT2_PID=$!

# Run client 3
export CLIENT_ID=3
export BOOTSTRAP_SERVERS=localhost:9094
python client.py &
CLIENT3_PID=$!

# Wait for clients to complete
echo "Waiting for clients to complete..."
wait $CLIENT1_PID
wait $CLIENT2_PID
wait $CLIENT3_PID

# Shutdown Kafka and server
echo "Shutting down Kafka and server..."
cd ../Server
docker-compose down

echo "Federated learning completed!"
