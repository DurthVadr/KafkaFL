#!/bin/bash

# Run Simple Federated Learning with 3 clients

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
SERVER_LOG="$LOG_DIR/server.log"
CLIENT1_LOG="$LOG_DIR/client1.log"
CLIENT2_LOG="$LOG_DIR/client2.log"
CLIENT3_LOG="$LOG_DIR/client3.log"

# Clean up any existing Kafka containers
echo "Cleaning up any existing Kafka containers..."
cd Server
docker-compose down

# Start Kafka
echo "Starting Kafka using docker-compose..."
docker-compose up -d kafka kafka-ui

# Wait for Kafka to start
echo "Waiting for Kafka to start..."
sleep 45  # Increased wait time

# Check if Kafka is running
echo "Checking if Kafka is accessible..."
for i in {1..10}; do
  if nc -z localhost 9094; then
    echo "Kafka is accessible on port 9094"
    break
  else
    echo "Waiting for Kafka to be accessible on port 9094 (attempt $i/10)..."
    sleep 10
  fi
done

# List Kafka topics
echo "Listing Kafka topics..."
docker exec -it server_kafka_1 kafka-topics.sh --bootstrap-server localhost:9092 --list || echo "Failed to list topics"

# Start the server
echo "Starting the server..."
cd ..
python Server/simple_server.py > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait for server to initialize
echo "Waiting for server to initialize..."
sleep 20  # Increased wait time

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "Server is running with PID $SERVER_PID"
else
    echo "Server failed to start. Check $SERVER_LOG for details."
    exit 1
fi

# Create topics manually if needed
echo "Creating topics manually..."
cd Server
docker exec -it server_kafka_1 kafka-topics.sh --bootstrap-server localhost:9092 --create --topic model_topic --partitions 1 --replication-factor 1 || echo "Topic model_topic may already exist"
docker exec -it server_kafka_1 kafka-topics.sh --bootstrap-server localhost:9092 --create --topic update_topic --partitions 1 --replication-factor 1 || echo "Topic update_topic may already exist"
cd ..

# Run 3 client instances
echo "Starting 3 client instances..."
cd Client

# Run client 1
export CLIENT_ID=1
export BOOTSTRAP_SERVERS=localhost:9094
python simple_client.py > $CLIENT1_LOG 2>&1 &
CLIENT1_PID=$!

# Wait a bit before starting the next client
sleep 5

# Run client 2
export CLIENT_ID=2
export BOOTSTRAP_SERVERS=localhost:9094
python simple_client.py > $CLIENT2_LOG 2>&1 &
CLIENT2_PID=$!

# Wait a bit before starting the next client
sleep 5

# Run client 3
export CLIENT_ID=3
export BOOTSTRAP_SERVERS=localhost:9094
python simple_client.py > $CLIENT3_LOG 2>&1 &
CLIENT3_PID=$!

# Check if clients are running
echo "Checking if clients are running..."
for pid in $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID; do
    if ps -p $pid > /dev/null; then
        echo "Client with PID $pid is running"
    else
        echo "Client with PID $pid failed to start"
    fi
done

# Wait for clients to complete or timeout after 5 minutes
echo "Waiting for clients to complete (max 5 minutes)..."
timeout 300 tail -f $CLIENT1_LOG $CLIENT2_LOG $CLIENT3_LOG &
TAIL_PID=$!

# Wait for clients with a timeout
timeout 300 bash -c "wait $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID" || {
    echo "Timeout waiting for clients to complete"
    kill $TAIL_PID 2>/dev/null
}

# Show summary of logs
echo "\n\n=== SERVER LOG SUMMARY ===\n"
tail -n 20 $SERVER_LOG

echo "\n\n=== CLIENT 1 LOG SUMMARY ===\n"
tail -n 20 $CLIENT1_LOG

echo "\n\n=== CLIENT 2 LOG SUMMARY ===\n"
tail -n 20 $CLIENT2_LOG

echo "\n\n=== CLIENT 3 LOG SUMMARY ===\n"
tail -n 20 $CLIENT3_LOG

# Kill the server
echo "Shutting down the server..."
kill $SERVER_PID 2>/dev/null

# Shutdown Kafka
echo "Shutting down Kafka..."
cd Server
docker-compose down

echo "Federated learning completed!"
echo "Check the logs in the $LOG_DIR directory for details."
