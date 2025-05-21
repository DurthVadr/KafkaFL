"""
Script to run the federated learning system with Kafka locally.
This approach uses less resources than Docker.
"""

import os
import subprocess
import time
import signal
import sys
import gc

# List to keep track of processes
processes = []

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("Shutting down all processes...")
    for process in processes:
        if process.poll() is None:  # If process is still running
            process.terminate()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_kafka_running():
    """Check if Kafka is running locally"""
    try:
        # Try to connect to Kafka
        from kafka.admin import KafkaAdminClient
        admin_client = KafkaAdminClient(bootstrap_servers='localhost:9092')
        admin_client.close()
        return True
    except Exception:
        return False

def start_kafka():
    """Start Kafka locally if not already running"""
    if check_kafka_running():
        print("Kafka is already running")
        return True

    print("Kafka is not running. Please start Kafka manually.")
    print("On macOS with Homebrew:")
    print("  1. Start Zookeeper: zookeeper-server-start /usr/local/etc/kafka/zookeeper.properties &")
    print("  2. Start Kafka: kafka-server-start /usr/local/etc/kafka/server.properties &")
    print("  3. Create topics:")
    print("     kafka-topics --create --bootstrap-server localhost:9092 --topic global_model --partitions 1 --replication-factor 1")
    print("     kafka-topics --create --bootstrap-server localhost:9092 --topic model_updates --partitions 1 --replication-factor 1")

    return False

def main():
    """Run the asynchronous federated learning system"""
    # Check if Kafka is running
    if not start_kafka():
        print("Please start Kafka and try again.")
        return

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Set environment variables to optimize resource usage
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
    os.environ["PYTHONUNBUFFERED"] = "1"      # Unbuffered Python output

    # Set time-based configuration with shorter durations for testing
    os.environ["DURATION_MINUTES"] = "30"                  # Run for 30 minutes
    os.environ["AGGREGATION_INTERVAL_SECONDS"] = "30"      # Aggregate every 30 seconds
    os.environ["MIN_UPDATES_PER_AGGREGATION"] = "1"        # Require at least 1 update
    os.environ["TRAINING_INTERVAL_SECONDS"] = "60"         # Train every 60 seconds

    # Start server
    print("Starting asynchronous server...")
    server_process = subprocess.Popen(
        ["python", "server.py"],
        env={**os.environ, "BOOTSTRAP_SERVERS": "localhost:9092"}
    )
    processes.append(server_process)

    # Wait for server to initialize
    print("Waiting for server to initialize...")
    time.sleep(10)

    # Start clients with staggered training intervals to avoid synchronization
    for i in range(1, 4):
        print(f"Starting client {i}...")
        # Stagger training intervals to avoid all clients training at the same time
        training_interval = 60 + (i * 10)  # 70, 80, 90 seconds
        client_process = subprocess.Popen(
            ["python", "client.py"],
            env={
                **os.environ,
                "BOOTSTRAP_SERVERS": "localhost:9092",
                "CLIENT_ID": str(i),
                "TRAINING_INTERVAL_SECONDS": str(training_interval)
            }
        )
        processes.append(client_process)
        time.sleep(2)  # Stagger client starts

    print("All processes started. Press Ctrl+C to stop.")

    # Wait for all processes to complete
    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    # Run garbage collection before starting
    gc.collect()

    try:
        main()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
