#!/bin/bash

# Default values
BROKER="localhost:9094"
NUM_CLIENTS=3
NUM_ROUNDS=3
USE_GRPC=false
MIN_CLIENTS=2

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --broker)
      BROKER="$2"
      shift 2
      ;;
    --num-clients)
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --num-rounds)
      NUM_ROUNDS="$2"
      shift 2
      ;;
    --grpc)
      USE_GRPC=true
      shift
      ;;
    --min-clients)
      MIN_CLIENTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Determine which protocol to use
if [ "$USE_GRPC" = true ]; then
  PROTOCOL_FLAG="--grpc"
  echo "Using gRPC for communication"
else
  PROTOCOL_FLAG=""
  echo "Using Kafka for communication"
fi

# Start the server
echo "Starting Flower server with broker $BROKER"
python flwr_implementation/server.py \
  --broker "$BROKER" \
  --num-rounds "$NUM_ROUNDS" \
  --min-clients "$MIN_CLIENTS" \
  --min-eval-clients "$MIN_CLIENTS" \
  --min-available-clients "$MIN_CLIENTS" \
  $PROTOCOL_FLAG > logs/server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Wait for server to initialize
sleep 5

# Start the clients
for ((i=1; i<=$NUM_CLIENTS; i++)); do
  echo "Starting client $i"
  python flwr_implementation/client.py \
    --broker "$BROKER" \
    --client-id "$i" \
    $PROTOCOL_FLAG > logs/client_$i.log 2>&1 &
  
  CLIENT_PID=$!
  echo "Client $i started with PID $CLIENT_PID"
  
  # Wait a bit between client starts to avoid overwhelming the server
  sleep 2
done

echo "All processes started. Check logs directory for output."
echo "Server log: logs/server.log"
echo "Client logs: logs/client_*.log"

# Wait for server to complete
wait $SERVER_PID
echo "Server process completed"
