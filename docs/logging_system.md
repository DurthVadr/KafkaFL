# Logging System

This document explains the logging system implemented in the federated learning project, which provides consistent and informative logging across server and client components.

## Overview

A robust logging system is essential for monitoring, debugging, and analyzing the behavior of distributed systems like federated learning. Our implementation provides colored console output for better readability and file-based logging for persistent records.

## Key Features

1. **Colored Console Output**: Different log levels are displayed with distinct colors for better visibility.
2. **File Logging**: All logs are saved to files for later analysis.
3. **Component-Specific Logs**: Server and clients have separate log files.
4. **Timestamp-Based Naming**: Log files include timestamps to track different runs.
5. **Configurable Log Levels**: Log verbosity can be adjusted as needed.

## Implementation

The logging system is implemented in `common/logger.py` and consists of the following components:

### 1. Custom Formatter with Colors

```python
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""
    
    LEVEL_COLORS = {
        logging.DEBUG: COLORS['BLUE'],
        logging.INFO: COLORS['GREEN'],
        logging.WARNING: COLORS['YELLOW'],
        logging.ERROR: COLORS['RED'],
        logging.CRITICAL: COLORS['PURPLE'],
    }
    
    def format(self, record):
        # Add colored level name
        levelname = record.levelname
        if record.levelno in self.LEVEL_COLORS:
            colored_levelname = f"{self.LEVEL_COLORS[record.levelno]}{levelname}{COLORS['RESET']}"
            record.colored_levelname = colored_levelname
        else:
            record.colored_levelname = levelname
            
        return super().format(record)
```

### 2. Logger Setup Function

```python
def setup_logger(name, log_level=logging.INFO, log_to_file=True, component_type=None):
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to a file (default: True)
        component_type: Type of component ('server', 'client', or None)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Use colored formatter for console
    console_formatter = ColoredFormatter(COLOR_LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create a log file with timestamp and component type
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        component_prefix = f"{component_type}_" if component_type else ""
        log_filename = f"logs/{component_prefix}{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        
        # Use standard formatter for file
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_filename}")
    
    return logger
```

### 3. Convenience Functions for Server and Clients

```python
def get_client_logger(client_id):
    """
    Get a logger configured for a client with the specified ID.
    
    Args:
        client_id: ID of the client
        
    Returns:
        Configured logger instance
    """
    return setup_logger(f"client_{client_id}", component_type="client")

def get_server_logger():
    """
    Get a logger configured for the server.
    
    Returns:
        Configured logger instance
    """
    return setup_logger("server", component_type="server")
```

## Usage

### Server Logging

```python
from common.logger import get_server_logger

# Initialize logger
self.logger = get_server_logger()

# Log messages
self.logger.info("Initializing federated learning server")
self.logger.warning("No updates received in round 1. Skipping aggregation.")
self.logger.error("Failed to send global model in round 2")
```

### Client Logging

```python
from common.logger import get_client_logger

# Initialize logger with client ID
self.logger = get_client_logger(self.client_id)

# Log messages
self.logger.info(f"Initializing client {self.client_id}")
self.logger.info(f"Training local model with {len(x_train)} samples")
self.logger.error("Failed to serialize model weights")
```

## Log File Structure

Log files are stored in the `logs` directory with the following naming convention:

- Server logs: `logs/server_server_YYYYMMDD_HHMMSS.log`
- Client logs: `logs/client_client_X_YYYYMMDD_HHMMSS.log` (where X is the client ID)

Each log entry includes:
- Timestamp
- Logger name
- Log level
- Message

Example:
```
2023-05-15 14:30:45,123 - server - INFO - Initializing federated learning server
2023-05-15 14:30:46,456 - server - INFO - Sending global model to clients
```

## Best Practices

1. **Use Appropriate Log Levels**:
   - DEBUG: Detailed information, typically useful only for diagnosing problems
   - INFO: Confirmation that things are working as expected
   - WARNING: Indication that something unexpected happened, but the application is still working
   - ERROR: Due to a more serious problem, the application has not been able to perform a function
   - CRITICAL: A serious error, indicating that the application itself may be unable to continue running

2. **Include Contextual Information**:
   - Include relevant IDs (client ID, round number, etc.)
   - Log input/output sizes rather than the actual data
   - Include timing information for performance-critical operations

3. **Log at Entry and Exit Points**:
   - Log at the beginning and end of important functions
   - Include success/failure status in exit logs

4. **Structured Logging**:
   - Use consistent message formats
   - Include key-value pairs for machine parsing

## Troubleshooting

### Common Issues

1. **Missing Logs**:
   - Check if the logs directory exists and is writable
   - Verify that the log level is appropriate (e.g., DEBUG messages won't appear if the log level is set to INFO)

2. **Duplicate Log Entries**:
   - This can happen if logger handlers are not properly cleared
   - Ensure `logger.handlers.clear()` is called before adding new handlers

3. **Performance Issues**:
   - Excessive logging can impact performance
   - Consider reducing log level in production environments
   - Use DEBUG level judiciously

## Future Improvements

1. **Structured Logging**: Implement JSON-formatted logs for better machine parsing.
2. **Log Rotation**: Add support for log rotation to manage file sizes.
3. **Remote Logging**: Add capability to send logs to a central server or logging service.
4. **Log Analysis Tools**: Develop tools to analyze and visualize logs.
5. **Context Managers**: Implement context managers for timing and tracking operations.

## Conclusion

The logging system is a critical component of the federated learning framework, providing visibility into the system's operation and facilitating debugging and performance analysis. The colored output enhances readability during development, while the file-based logging ensures that a complete record is available for later analysis.
