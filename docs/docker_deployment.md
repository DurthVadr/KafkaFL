# Docker Deployment

## Overview

This guide covers deploying the federated learning system using Docker containers. The system provides pre-built Docker images for both server and client components, along with Docker Compose configurations for easy orchestration.

## Docker Architecture

### Container Components

The Docker deployment consists of the following containers:

1. **Kafka Broker**: Message broker for federated learning communication
2. **Zookeeper**: Coordination service for Kafka
3. **FL Server**: Federated learning coordination server
4. **FL Clients**: Multiple federated learning client instances
5. **Monitoring** (Optional): Prometheus and Grafana for system monitoring

### Network Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FL Client 1   │    │   FL Client 2   │    │   FL Client N   │
│                 │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │      Kafka Broker       │
                    │                         │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────┴───────────┐
                    │      FL Server          │
                    │                         │
                    └─────────────────────────┘
```

## Prerequisites

### System Requirements

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Memory**: At least 4GB RAM (8GB recommended for multiple clients)
- **CPU**: 2+ cores recommended
- **Storage**: 10GB free space minimum

### Installation Verification

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Verify Docker is running
docker info
```

## Quick Start with Docker Compose

### 1. Basic Deployment

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd cs402

# Start the entire system
docker-compose up -d

# Check running containers
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Scaled Deployment

```bash
# Start with multiple clients
docker-compose up -d --scale fl-client=5

# Start with custom configuration
docker-compose -f docker-compose.prod.yml up -d
```

## Docker Compose Configurations

### Development Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    hostname: zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper-data:/var/lib/zookeeper/data
      - zookeeper-logs:/var/lib/zookeeper/log

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    hostname: kafka
    container_name: kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9997:9997"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_JMX_PORT: 9997
      KAFKA_JMX_HOSTNAME: localhost
    volumes:
      - kafka-data:/var/lib/kafka/data

  fl-server:
    build:
      context: .
      dockerfile: Dockerfile.server
    container_name: fl-server
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - FL_SERVER_ID=server-001
      - FL_MIN_CLIENTS=2
      - FL_MAX_ROUNDS=10
    volumes:
      - ./logs:/app/logs
      - ./plots:/app/plots
    command: ["python", "server.py", "--kafka-servers", "kafka:29092"]

  fl-client:
    build:
      context: .
      dockerfile: Dockerfile.client
    depends_on:
      - kafka
      - fl-server
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - FL_CLIENT_ID=client-${HOSTNAME}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    command: ["python", "client.py", "--kafka-servers", "kafka:29092", "--client-id", "client-${HOSTNAME}"]
    deploy:
      replicas: 2

volumes:
  zookeeper-data:
  zookeeper-logs:
  kafka-data:

networks:
  default:
    name: fl-network
```

### Production Configuration (`docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    hostname: zookeeper
    container_name: zookeeper
    restart: unless-stopped
    ports:
      - "2181:2181"
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
      ZOOKEEPER_SYNC_LIMIT: 2
      ZOOKEEPER_INIT_LIMIT: 5
    volumes:
      - zookeeper-data:/var/lib/zookeeper/data
      - zookeeper-logs:/var/lib/zookeeper/log
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    hostname: kafka
    container_name: kafka
    restart: unless-stopped
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: 'zookeeper:2181'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_NUM_PARTITIONS: 3
      KAFKA_DEFAULT_REPLICATION_FACTOR: 1
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      KAFKA_LOG_RETENTION_CHECK_INTERVAL_MS: 300000
    volumes:
      - kafka-data:/var/lib/kafka/data
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  fl-server:
    build:
      context: .
      dockerfile: Dockerfile.server
    container_name: fl-server
    restart: unless-stopped
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - FL_SERVER_ID=server-001
      - FL_MIN_CLIENTS=3
      - FL_MAX_ROUNDS=50
      - FL_ROUND_TIMEOUT=600
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./plots:/app/plots
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"

  fl-client:
    build:
      context: .
      dockerfile: Dockerfile.client
    restart: unless-stopped
    depends_on:
      - kafka
      - fl-server
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - FL_CLIENT_ID=client-${RANDOM_ID}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: "1"
        reservations:
          memory: 1G
          cpus: "0.5"
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  zookeeper-data:
  zookeeper-logs:
  kafka-data:

networks:
  default:
    name: fl-network
```

## Dockerfile Configurations

### Server Dockerfile (`Dockerfile.server`)

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY common/ ./common/
COPY flower_implementation/ ./flower_implementation/

# Create necessary directories
RUN mkdir -p logs plots models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "server.py"]
```

### Client Dockerfile (`Dockerfile.client`)

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY client.py .
COPY common/ ./common/
COPY flower_implementation/ ./flower_implementation/

# Create necessary directories
RUN mkdir -p logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "client.py"]
```

## Advanced Deployment Scenarios

### Multi-Host Deployment

For deploying across multiple hosts, use Docker Swarm:

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.swarm.yml fl-stack

# Scale services
docker service scale fl-stack_fl-client=10
```

### Kubernetes Deployment

Basic Kubernetes deployment configuration:

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: federated-learning

---
# kubernetes/kafka-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kafka
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
    spec:
      containers:
      - name: kafka
        image: confluentinc/cp-kafka:7.4.0
        ports:
        - containerPort: 9092
        env:
        - name: KAFKA_ZOOKEEPER_CONNECT
          value: "zookeeper:2181"
        # ... other environment variables

---
# kubernetes/fl-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-server
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-server
  template:
    metadata:
      labels:
        app: fl-server
    spec:
      containers:
      - name: fl-server
        image: fl-server:latest
        ports:
        - containerPort: 8080
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
```

### GPU-Enabled Deployment

For GPU-accelerated training:

```yaml
# docker-compose.gpu.yml
version: '3.8'

services:
  fl-client-gpu:
    build:
      context: .
      dockerfile: Dockerfile.client.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_VISIBLE_DEVICES=all
```

## Monitoring and Observability

### Adding Prometheus and Grafana

```yaml
# monitoring/docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  prometheus-data:
  grafana-data:
```

### Kafka Monitoring

```yaml
# Add to main docker-compose.yml
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    depends_on:
      - kafka
    ports:
      - "8081:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:29092
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
```

## Management Commands

### Container Management

```bash
# View running containers
docker-compose ps

# Check container logs
docker-compose logs fl-server
docker-compose logs fl-client

# Follow logs in real-time
docker-compose logs -f

# Restart specific service
docker-compose restart fl-server

# Scale client services
docker-compose up -d --scale fl-client=5

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Debugging Commands

```bash
# Execute command in running container
docker-compose exec fl-server bash
docker-compose exec fl-client python -c "import torch; print(torch.cuda.is_available())"

# Check container resource usage
docker stats

# Inspect container configuration
docker inspect fl-server

# View container filesystem
docker-compose exec fl-server ls -la /app
```

### Backup and Recovery

```bash
# Backup volumes
docker run --rm -v cs402_kafka-data:/data -v $(pwd):/backup alpine \
    tar czf /backup/kafka-data-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v cs402_kafka-data:/data -v $(pwd):/backup alpine \
    tar xzf /backup/kafka-data-backup.tar.gz -C /data

# Export container as image
docker commit fl-server fl-server:backup
docker save fl-server:backup > fl-server-backup.tar
```

## Security Considerations

### Network Security

```yaml
# Secure network configuration
networks:
  fl-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Secrets Management

```yaml
# Using Docker secrets
services:
  fl-server:
    secrets:
      - kafka_password
      - ssl_cert
    environment:
      - KAFKA_PASSWORD_FILE=/run/secrets/kafka_password

secrets:
  kafka_password:
    file: ./secrets/kafka_password.txt
  ssl_cert:
    file: ./secrets/ssl_cert.pem
```

### SSL/TLS Configuration

```yaml
# SSL-enabled Kafka
  kafka:
    environment:
      KAFKA_SSL_KEYSTORE_FILENAME: kafka.server.keystore.jks
      KAFKA_SSL_KEYSTORE_CREDENTIALS: kafka_keystore_creds
      KAFKA_SSL_KEY_CREDENTIALS: kafka_ssl_key_creds
      KAFKA_SSL_TRUSTSTORE_FILENAME: kafka.server.truststore.jks
      KAFKA_SSL_TRUSTSTORE_CREDENTIALS: kafka_truststore_creds
    volumes:
      - ./ssl:/etc/kafka/secrets
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**
   ```bash
   # Check logs
   docker-compose logs <service-name>
   
   # Check Docker daemon
   docker info
   
   # Verify image build
   docker build -t fl-server -f Dockerfile.server .
   ```

2. **Network Connectivity Issues**
   ```bash
   # Test network connectivity
   docker-compose exec fl-client ping kafka
   
   # Check network configuration
   docker network ls
   docker network inspect cs402_default
   ```

3. **Resource Constraints**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Adjust resource limits
   docker-compose up -d --scale fl-client=2
   ```

4. **Volume Permission Issues**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ./logs ./data
   ```

### Performance Tuning

```bash
# Optimize Docker daemon
echo '{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}' | sudo tee /etc/docker/daemon.json

# Restart Docker daemon
sudo systemctl restart docker
```

This comprehensive Docker deployment guide provides everything needed to deploy and manage the federated learning system in containerized environments, from development to production scenarios.
