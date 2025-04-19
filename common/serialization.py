"""
Serialization utilities for federated learning system.
Provides functions for serializing and deserializing model weights.
"""

import io
import logging
import hashlib
import numpy as np
import traceback

def serialize_weights(weights, logger=None):
    """
    Serialize model weights to bytes.

    Args:
        weights: List of numpy arrays representing model weights
        logger: Logger instance for logging (optional)

    Returns:
        Serialized weights as bytes, or None if serialization fails
    """
    try:
        # Create a BytesIO buffer
        buffer = io.BytesIO()

        # Save the number of arrays
        buffer.write(np.array([len(weights)], dtype=np.int32).tobytes())

        # Save each array with its shape and type information
        for i, arr in enumerate(weights):
            # Save array index
            buffer.write(np.array([i], dtype=np.int32).tobytes())

            # Save data type information
            dtype_str = str(arr.dtype).encode('utf-8')
            buffer.write(np.array([len(dtype_str)], dtype=np.int32).tobytes())
            buffer.write(dtype_str)

            # Save shape information
            shape = np.array(arr.shape, dtype=np.int32)
            buffer.write(np.array([len(shape)], dtype=np.int32).tobytes())
            buffer.write(shape.tobytes())

            # Save array data
            arr_bytes = arr.tobytes()
            buffer.write(np.array([len(arr_bytes)], dtype=np.int32).tobytes())
            buffer.write(arr_bytes)

        # Get the serialized data
        serialized_weights = buffer.getvalue()
        buffer.close()

        # Calculate checksum
        checksum = hashlib.md5(serialized_weights).hexdigest()

        # Log information if logger is provided
        if logger:
            logger.info(f"Serialized {len(weights)} weight arrays, size: {len(serialized_weights)} bytes")
            logger.debug(f"Serialized data checksum: {checksum}")

        return serialized_weights
    except Exception as e:
        if logger:
            logger.error(f"Error serializing weights: {e}")
            logger.error(traceback.format_exc())
        return None

def deserialize_weights(buffer, logger=None):
    """
    Deserialize model weights from bytes.

    Args:
        buffer: Serialized weights as bytes
        logger: Logger instance for logging (optional)

    Returns:
        List of numpy arrays representing model weights, or None if deserialization fails
    """
    try:
        # Calculate checksum for verification
        checksum = hashlib.md5(buffer).hexdigest()
        if logger:
            logger.debug(f"Received data checksum: {checksum}")

        # Create a BytesIO buffer from the received bytes
        buffer_io = io.BytesIO(buffer)

        # Read the number of arrays
        num_arrays = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]
        if logger:
            logger.info(f"Deserializing {num_arrays} weight arrays")

        # Read each array
        weights = [None] * num_arrays
        for _ in range(num_arrays):
            try:
                # Read array index
                array_idx = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]

                # Read data type information
                dtype_len = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]
                dtype_str = buffer_io.read(dtype_len).decode('utf-8')
                dtype = np.dtype(dtype_str)

                # Read shape information
                ndim = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]
                shape = tuple(np.frombuffer(buffer_io.read(4 * ndim), dtype=np.int32))

                # Read array data size
                data_size = np.frombuffer(buffer_io.read(4), dtype=np.int32)[0]

                # Read array data
                arr_data = np.frombuffer(buffer_io.read(data_size), dtype=dtype).reshape(shape)

                # Store in the correct position
                weights[array_idx] = arr_data

                if logger:
                    logger.debug(f"Deserialized weight {array_idx} with shape {shape} and dtype {dtype}")
            except Exception as e:
                if logger:
                    logger.error(f"Error deserializing weight array {_}: {e}")
                    logger.error(traceback.format_exc())

        buffer_io.close()

        # Check if all weights were deserialized correctly
        missing_indices = [i for i, w in enumerate(weights) if w is None]
        if missing_indices:
            if logger:
                logger.error(f"Missing weight arrays at indices: {missing_indices}")
            return None

        if logger:
            logger.info(f"Successfully deserialized {len(weights)} weight arrays")

        return weights
    except Exception as e:
        if logger:
            logger.error(f"Error deserializing weights: {e}")
            logger.error(traceback.format_exc())
        return None
