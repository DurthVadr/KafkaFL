"""
Logging module for federated learning system.
Provides consistent logging across server and client components.
"""

import logging
import os
import sys
from datetime import datetime

# Define log levels with colors for better visibility
COLORS = {
    'RESET': '\033[0m',
    'RED': '\033[91m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'BLUE': '\033[94m',
    'PURPLE': '\033[95m',
    'CYAN': '\033[96m',
}

# Custom log format with colors
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
COLOR_LOG_FORMAT = '%(asctime)s - %(name)s - %(colored_levelname)s - %(message)s'

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
