"""
Check what's available in the flwr.common module.
"""

import logging
logging.basicConfig(level=logging.INFO)

try:
    import flwr
    logging.info(f"Flower version: {flwr.__version__}")
    
    import flwr.common
    logging.info(f"Available in flwr.common: {dir(flwr.common)}")
    
    # Check if we can import Parameters
    from flwr.common import Parameters
    logging.info("Successfully imported Parameters")
    
    # Check if we can create a Parameters object
    params = Parameters(tensors=[], tensor_type="numpy.ndarray")
    logging.info(f"Created Parameters object: {params}")
    
except Exception as e:
    logging.error(f"Error: {e}")
