"""
Kafka utilities for federated learning system.
Provides functions for connecting to Kafka and sending/receiving messages.
"""

import time
import logging
import traceback
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

def create_producer(bootstrap_servers, logger=None):
    """
    Create a Kafka producer with appropriate configuration.

    Args:
        bootstrap_servers: Kafka bootstrap servers
        logger: Logger instance for logging (optional)

    Returns:
        KafkaProducer instance, or None if creation fails
    """
    max_attempts = 5
    attempt = 0
    retry_delay = 5

    while attempt < max_attempts:
        try:
            # Create Kafka producer with appropriate configuration
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: v,  # Keep raw bytes
                max_request_size=20971520,  # 20MB max message size
                buffer_memory=41943040,  # 40MB buffer memory
                compression_type='gzip'  # Enable compression
            )

            if logger:
                logger.info(f"Successfully created Kafka producer connected to {bootstrap_servers}")

            return producer
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create Kafka producer (attempt {attempt+1}/{max_attempts}): {e}")

            attempt += 1
            if attempt < max_attempts:
                if logger:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    if logger:
        logger.error("Failed to create Kafka producer after multiple attempts")

    return None

def create_consumer(bootstrap_servers, group_id, topics, logger=None):
    """
    Create a Kafka consumer with appropriate configuration.

    Args:
        bootstrap_servers: Kafka bootstrap servers
        group_id: Consumer group ID
        topics: List of topics to subscribe to
        logger: Logger instance for logging (optional)

    Returns:
        KafkaConsumer instance, or None if creation fails
    """
    max_attempts = 5
    attempt = 0
    retry_delay = 5

    while attempt < max_attempts:
        try:
            # Create Kafka consumer with appropriate configuration
            consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda m: m,  # Keep raw bytes
                max_partition_fetch_bytes=10485760,  # 10MB max fetch size
                fetch_max_bytes=52428800  # 50MB max fetch bytes
            )

            # Subscribe to topics
            consumer.subscribe(topics)

            if logger:
                logger.info(f"Successfully created Kafka consumer connected to {bootstrap_servers}")
                logger.info(f"Subscribed to topics: {topics}")

            return consumer
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create Kafka consumer (attempt {attempt+1}/{max_attempts}): {e}")

            attempt += 1
            if attempt < max_attempts:
                if logger:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    if logger:
        logger.error("Failed to create Kafka consumer after multiple attempts")

    return None

def send_message(producer, topic, message, logger=None):
    """
    Send a message to a Kafka topic.

    Args:
        producer: KafkaProducer instance
        topic: Topic to send the message to
        message: Message to send (bytes)
        logger: Logger instance for logging (optional)

    Returns:
        True if the message was sent successfully, False otherwise
    """
    if producer is None:
        if logger:
            logger.error("Cannot send message: producer is None")
        return False

    try:
        # Send the message
        future = producer.send(topic, message)
        record_metadata = future.get(timeout=10)
        producer.flush()

        if logger:
            logger.info(f"Message sent to topic: {record_metadata.topic}, "
                       f"partition: {record_metadata.partition}, "
                       f"offset: {record_metadata.offset}")

        return True
    except Exception as e:
        if logger:
            logger.error(f"Error sending message to topic {topic}: {e}")
            logger.error(traceback.format_exc())

        return False

def receive_messages(consumer, timeout_ms=60000, max_messages=1, logger=None):
    """
    Receive messages from Kafka topics.

    Args:
        consumer: KafkaConsumer instance
        timeout_ms: Timeout in milliseconds (default: 60000)
        max_messages: Maximum number of messages to receive (default: 1)
        logger: Logger instance for logging (optional)

    Returns:
        List of received messages, or empty list if no messages were received
    """
    if consumer is None:
        if logger:
            logger.error("Cannot receive messages: consumer is None")
        return []

    try:
        messages = []
        start_time = time.time()
        poll_count = 0

        while len(messages) < max_messages and (time.time() - start_time) < (timeout_ms/1000):
            # Poll for messages
            poll_result = consumer.poll(timeout_ms=5000)
            poll_count += 1

            if poll_result:
                if logger:
                    logger.debug(f"Poll attempt {poll_count}, received {sum(len(msgs) for msgs in poll_result.values())} messages")

                # Process received messages
                for tp, msgs in poll_result.items():
                    if logger:
                        logger.debug(f"Processing messages from topic-partition: {tp.topic}-{tp.partition}")

                    for msg in msgs:
                        messages.append(msg.value)

                        if len(messages) >= max_messages:
                            break

                    if len(messages) >= max_messages:
                        break
            else:
                if logger:
                    logger.debug(f"Poll attempt {poll_count}: No messages received")

                # Sleep to avoid tight polling
                time.sleep(1)

        if logger:
            logger.info(f"Received {len(messages)} messages after {poll_count} poll attempts")

        return messages
    except Exception as e:
        if logger:
            logger.error(f"Error receiving messages: {e}")
            logger.error(traceback.format_exc())

        return []

def close_kafka_resources(producer=None, consumer=None, logger=None):
    """
    Close Kafka producer and consumer resources.

    Args:
        producer: KafkaProducer instance (optional)
        consumer: KafkaConsumer instance (optional)
        logger: Logger instance for logging (optional)
    """
    # Close consumer
    if consumer is not None:
        try:
            consumer.unsubscribe()
            consumer.close()
            if logger:
                logger.info("Kafka consumer closed")
        except Exception as e:
            if logger:
                logger.error(f"Error closing Kafka consumer: {e}")

    # Close producer
    if producer is not None:
        try:
            producer.flush()
            producer.close()
            if logger:
                logger.info("Kafka producer closed")
        except Exception as e:
            if logger:
                logger.error(f"Error closing Kafka producer: {e}")
