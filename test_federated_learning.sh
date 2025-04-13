#!/bin/bash

# Test script for federated learning with Kafka

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
SERVER_LOG="$LOG_DIR/server.log"
CLIENT1_LOG="$LOG_DIR/client1.log"
CLIENT2_LOG="$LOG_DIR/client2.log"
CLIENT3_LOG="$LOG_DIR/client3.log"

# Make sure the log directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

# Clean up any existing Kafka containers
echo "Cleaning up any existing Kafka containers..."
cd Server
docker-compose down

# Start Kafka
echo "Starting Kafka using docker-compose..."
docker-compose up -d kafka kafka-ui

# Wait for Kafka to start
echo "Waiting for Kafka to start..."
sleep 30

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

# Create topics manually
echo "Creating topics manually..."
docker exec -it server-kafka-1 kafka-topics.sh --bootstrap-server localhost:9092 --create --topic model_topic --partitions 1 --replication-factor 1 || echo "Topic model_topic may already exist"
docker exec -it server-kafka-1 kafka-topics.sh --bootstrap-server localhost:9092 --create --topic update_topic --partitions 1 --replication-factor 1 || echo "Topic update_topic may already exist"

# List Kafka topics
echo "Listing Kafka topics..."
docker exec -it server-kafka-1 kafka-topics.sh --bootstrap-server localhost:9092 --list

# Start the server
echo "Starting the server..."
cd ..
export BOOTSTRAP_SERVERS=localhost:9094
python Server/server.py > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# Wait for server to initialize
echo "Waiting for server to initialize..."
sleep 20

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo "Server is running with PID $SERVER_PID"
else
    echo "Server failed to start. Check $SERVER_LOG for details."
    exit 1
fi

# Run 3 client instances
echo "Starting 3 client instances..."
cd Client

# Run client 1
export CLIENT_ID=1
export BOOTSTRAP_SERVERS=localhost:9094
python client.py > $CLIENT1_LOG 2>&1 &
CLIENT1_PID=$!

# Wait a bit before starting the next client
sleep 5

# Run client 2
export CLIENT_ID=2
export BOOTSTRAP_SERVERS=localhost:9094
python client.py > $CLIENT2_LOG 2>&1 &
CLIENT2_PID=$!

# Wait a bit before starting the next client
sleep 5

# Run client 3
export CLIENT_ID=3
export BOOTSTRAP_SERVERS=localhost:9094
python client.py > $CLIENT3_LOG 2>&1 &
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

# Wait for clients to complete (with timeout)
echo "Waiting for clients to complete (max 5 minutes)..."

# Function to wait for a process with timeout
wait_with_timeout() {
    local pid=$1
    local timeout=300  # 5 minutes
    local count=0
    while kill -0 $pid 2>/dev/null && [ $count -lt $timeout ]; do
        sleep 1
        count=$((count+1))
    done
    if [ $count -ge $timeout ]; then
        echo "Timeout waiting for process $pid"
        kill -9 $pid 2>/dev/null
        return 1
    fi
    return 0
}

# Wait for each client with timeout
wait_with_timeout $CLIENT1_PID
wait_with_timeout $CLIENT2_PID
wait_with_timeout $CLIENT3_PID

# Kill the server
echo "Shutting down the server..."
kill $SERVER_PID 2>/dev/null

# Show summary of logs
echo -e "\n\n=== SERVER LOG SUMMARY ===\n"
tail -n 20 $SERVER_LOG

echo -e "\n\n=== CLIENT 1 LOG SUMMARY ===\n"
tail -n 20 $CLIENT1_LOG

echo -e "\n\n=== CLIENT 2 LOG SUMMARY ===\n"
tail -n 20 $CLIENT2_LOG

echo -e "\n\n=== CLIENT 3 LOG SUMMARY ===\n"
tail -n 20 $CLIENT3_LOG

# Shutdown Kafka
echo "Shutting down Kafka..."
cd Server
docker-compose down

echo "Federated learning test completed!"
echo "Check the logs in the $LOG_DIR directory for details."
