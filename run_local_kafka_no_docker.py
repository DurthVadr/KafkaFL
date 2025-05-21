"""
Script to run the federated learning system with Kafka locally without Docker.
This approach uses minimal system resources and doesn't require Docker.

It uses the kafka-python library to check if Kafka is running and provides
instructions for installing and running Kafka locally if needed.
"""

import os
import subprocess
import time
import signal
import sys
import gc
import socket
import atexit
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# List to keep track of processes
processes = []

def signal_handler(sig, frame):
    """Handle termination signals"""
    print("\nShutting down all processes...")
    cleanup_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def cleanup_resources():
    """Clean up all resources"""
    # Stop all child processes
    for process in processes:
        if process.poll() is None:  # If process is still running
            try:
                process.terminate()
                print(f"Terminated process with PID {process.pid}")
            except Exception as e:
                print(f"Error terminating process: {e}")

# Register cleanup function to run on exit
atexit.register(cleanup_resources)

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def check_kafka_running():
    """Check if Kafka is running locally"""
    # Check if port 9092 is in use (Kafka's default port)
    if is_port_in_use(9092):
        try:
            # Try to connect to Kafka using IPv4
            admin_client = KafkaAdminClient(bootstrap_servers='127.0.0.1:9092')
            admin_client.close()
            return True
        except Exception as e:
            # Port is in use but not by Kafka
            print(f"Port 9092 is in use but doesn't seem to be Kafka: {e}")
            return False
    return False

def create_kafka_topics():
    """Create Kafka topics needed for federated learning"""
    try:
        print("Creating Kafka topics...")
        admin_client = KafkaAdminClient(bootstrap_servers='127.0.0.1:9092')

        # Define topics
        topics = [
            NewTopic(name="global_model", num_partitions=1, replication_factor=1),
            NewTopic(name="model_updates", num_partitions=1, replication_factor=1)
        ]

        # Create topics
        for topic in topics:
            try:
                admin_client.create_topics([topic])
                print(f"Created topic: {topic.name}")
            except TopicAlreadyExistsError:
                print(f"Topic already exists: {topic.name}")

        admin_client.close()
        return True
    except Exception as e:
        print(f"Error creating Kafka topics: {e}")
        return False

def start_kafka_with_custom_config():
    """Start Kafka with custom configuration for large messages"""
    try:
        print("Starting Kafka with custom configuration...")

        # Stop Kafka if it's running
        subprocess.run(["brew", "services", "stop", "kafka"], check=False)
        time.sleep(3)

        # Start Zookeeper if it's not running
        subprocess.run(["brew", "services", "start", "zookeeper"], check=False)
        time.sleep(3)

        # Start Kafka using Homebrew services
        subprocess.run(["brew", "services", "start", "kafka"], check=True)

        # Wait for Kafka to start
        print("Waiting for Kafka to start...")
        start_time = time.time()
        while time.time() - start_time < 30:  # Wait up to 30 seconds
            if check_kafka_running():
                print("Kafka is now running")

                # Now we need to update the Kafka server.properties to allow large messages
                print("Updating Kafka configuration for large messages...")

                # Find the Kafka server.properties file
                kafka_config_path = "/opt/homebrew/etc/kafka/server.properties"
                if not os.path.exists(kafka_config_path):
                    # Try alternative path
                    kafka_config_path = "/usr/local/etc/kafka/server.properties"
                    if not os.path.exists(kafka_config_path):
                        print("Could not find Kafka server.properties file")
                        return False

                # Read the current configuration
                with open(kafka_config_path, 'r') as f:
                    config = f.read()

                # Add or update message size limits
                config_updates = {
                    "message.max.bytes": "104857600",  # 100MB
                    "replica.fetch.max.bytes": "104857600",  # 100MB
                    "max.request.size": "104857600",  # 100MB
                    "socket.request.max.bytes": "104857600"  # 100MB
                }

                # Apply updates
                for key, value in config_updates.items():
                    if f"{key}=" in config:
                        # Update existing property
                        lines = config.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip().startswith(f"{key}="):
                                lines[i] = f"{key}={value}"
                                break
                        config = '\n'.join(lines)
                    else:
                        # Add new property
                        config += f"\n{key}={value}"

                # Write the updated configuration
                with open(kafka_config_path, 'w') as f:
                    f.write(config)

                # Restart Kafka to apply changes
                print("Restarting Kafka to apply configuration changes...")
                subprocess.run(["brew", "services", "restart", "kafka"], check=True)

                # Wait for Kafka to restart
                print("Waiting for Kafka to restart...")
                time.sleep(5)

                # Check if Kafka is running after restart
                start_time = time.time()
                while time.time() - start_time < 30:  # Wait up to 30 seconds
                    if check_kafka_running():
                        print("Kafka is now running with large message configuration")
                        # Wait a bit more to ensure Kafka is fully initialized
                        time.sleep(5)
                        return True
                    time.sleep(1)

                print("Kafka didn't restart properly. Please check Kafka logs.")
                return False

            time.sleep(1)

        print("Kafka didn't start within the expected time. Please check Kafka logs.")
        return False
    except Exception as e:
        print(f"Error starting Kafka: {e}")
        return False

def main():
    """Run the asynchronous federated learning system"""
    print("=== Federated Learning with Kafka (No Docker) ===")

    # Check if Kafka is already running
    if check_kafka_running():
        print("Kafka is already running on localhost:9092")
        print("Restarting Kafka with custom configuration...")
        if not start_kafka_with_custom_config():
            print("Failed to restart Kafka with custom configuration.")
            print("Please restart it manually with: brew services restart kafka")
            return
    else:
        print("Kafka is not running. Starting it with custom configuration...")
        if not start_kafka_with_custom_config():
            print("Failed to start Kafka with custom configuration.")
            print("Please start it manually with: brew services start kafka")
            return

    # Create topics
    if not create_kafka_topics():
        print("Failed to create Kafka topics. Please check if Kafka is properly configured.")
        return

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Set environment variables to optimize resource usage
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
    os.environ["PYTHONUNBUFFERED"] = "1"      # Unbuffered Python output
    os.environ["REDUCED_DATA_SIZE"] = "1"     # Use reduced data size

    # Set time-based configuration with shorter durations for testing
    os.environ["DURATION_MINUTES"] = "30"                  # Run for 30 minutes
    os.environ["AGGREGATION_INTERVAL_SECONDS"] = "30"      # Aggregate every 30 seconds
    os.environ["MIN_UPDATES_PER_AGGREGATION"] = "1"        # Require at least 1 update
    os.environ["TRAINING_INTERVAL_SECONDS"] = "60"         # Train every 60 seconds

    # Use IPv4 for Kafka connections
    os.environ["KAFKA_OPTS"] = "-Djava.net.preferIPv4Stack=true"

    # Start server
    print("Starting asynchronous server...")
    server_process = subprocess.Popen(
        [sys.executable, "server.py"],
        env={**os.environ, "BOOTSTRAP_SERVERS": "127.0.0.1:9092"}
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
            [sys.executable, "client.py"],
            env={
                **os.environ,
                "BOOTSTRAP_SERVERS": "127.0.0.1:9092",
                "CLIENT_ID": str(i),
                "TRAINING_INTERVAL_SECONDS": str(training_interval)
            }
        )
        processes.append(client_process)
        time.sleep(2)  # Stagger client starts

    print("\nAll processes started. Press Ctrl+C to stop.")
    print("The system will run for approximately 30 minutes.")
    print("You can monitor the logs in the 'logs' directory.")

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
    finally:
        cleanup_resources()
