"""
Model definition module for federated learning system.
Provides consistent model architecture across server and clients.
"""

import logging
import numpy as np

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    # Configure TensorFlow for CPU-only operation
    tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available in model.py")
    TENSORFLOW_AVAILABLE = False

# Current model version
MODEL_VERSION = "v2"

def create_lenet_model():
    """
    Create a LeNet model for CIFAR-10 classification.

    LeNet is a classic convolutional neural network architecture
    designed by Yann LeCun in the 1990s, originally for handwritten digit recognition.

    Returns:
        A compiled Keras model if TensorFlow is available, None otherwise
    """
    if not TENSORFLOW_AVAILABLE:
        logging.warning("TensorFlow not available. Cannot create LeNet model.")
        return None

    try:
        # LeNet architecture adapted for CIFAR-10
        logging.info("Creating LeNet model for CIFAR-10")
        model = Sequential([
            # First convolutional layer
            Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
            AveragePooling2D(pool_size=(2, 2)),

            # Second convolutional layer
            Conv2D(16, kernel_size=(5, 5), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),

            # Fully connected layers
            Flatten(),
            Dense(120, activation='relu'),
            Dense(84, activation='relu'),
            Dense(10, activation='softmax')
        ])

        # Compile model with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        logging.error(f"Error creating LeNet model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_cifar10_model():
    """
    Create a model for CIFAR-10 classification.

    Returns:
        A compiled Keras model if TensorFlow is available, None otherwise
    """
    if not TENSORFLOW_AVAILABLE:
        logging.warning("TensorFlow not available. Cannot create model.")
        return None

    try:
        # Standard CNN for CIFAR-10
        logging.info("Using standard model for CIFAR-10")
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            # Second convolutional block
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            # Third convolutional block
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),

            # Fully connected layers
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])

        # Compile model with Adam optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        logging.error(f"Error creating CIFAR-10 model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def get_random_weights(model_type="standard"):
    """
    Generate random weights for the model when TensorFlow is not available.

    Args:
        model_type: Type of model to generate weights for ("standard" or "lenet")

    Returns:
        List of numpy arrays with random weights
    """
    if TENSORFLOW_AVAILABLE:
        # Create model and get its weight shapes
        if model_type.lower() == "lenet":
            model = create_lenet_model()
        else:
            model = create_cifar10_model()

        if model is None:
            return [np.random.rand(10).astype(np.float32)]

        # Initialize with random weights
        weights = [np.random.normal(0, 0.05, w.shape).astype(np.float32) for w in model.get_weights()]
        return weights
    else:
        if model_type.lower() == "lenet":
            # Approximate shapes for the LeNet model
            shapes = [
                # First conv layer
                (5, 5, 3, 6), (6,),  # Conv2D

                # Second conv layer
                (5, 5, 6, 16), (16,),  # Conv2D

                # Fully connected layers
                (400, 120), (120,),  # Dense (flattened 5x5x16 to 120)
                (120, 84), (84,),    # Dense
                (84, 10), (10,)      # Output layer
            ]
        else:
            # Approximate shapes for the standard model defined in create_cifar10_model
            shapes = [
                # First conv block
                (3, 3, 3, 32), (32,), (32,), (32,), (32,), (32,),  # Conv2D + BN
                (3, 3, 32, 32), (32,), (32,), (32,), (32,), (32,),  # Conv2D + BN

                # Second conv block
                (3, 3, 32, 64), (64,), (64,), (64,), (64,), (64,),  # Conv2D + BN
                (3, 3, 64, 64), (64,), (64,), (64,), (64,), (64,),  # Conv2D + BN

                # Third conv block
                (3, 3, 64, 128), (128,), (128,), (128,), (128,), (128,),  # Conv2D + BN
                (3, 3, 128, 128), (128,), (128,), (128,), (128,), (128,),  # Conv2D + BN

                # Fully connected layers
                (2048, 128), (128,), (128,), (128,), (128,), (128,),  # Dense + BN
                (128, 10), (10,)  # Output layer
            ]

        # Create random weights with the appropriate shapes
        weights = [np.random.normal(0, 0.05, shape).astype(np.float32) for shape in shapes]
        return weights

def are_weights_compatible(model, weights):
    """
    Check if weights are compatible with the model.

    Args:
        model: Keras model
        weights: List of weight arrays

    Returns:
        Boolean indicating compatibility
    """
    if model is None:
        return False

    model_weights = model.get_weights()

    # Log the number of weight arrays for debugging
    logging.info(f"Model has {len(model_weights)} weight arrays, received {len(weights)} weight arrays")

    # For LeNet, we might receive weights from a different model
    # If the model is LeNet (which has 10 weight arrays), try to use just the first 10 arrays
    if len(model_weights) == 10 and len(weights) > 10:
        logging.info("LeNet model detected with more weights than expected. Will try to adapt.")
        return False

    # Check if number of weight arrays matches
    if len(model_weights) != len(weights):
        logging.warning(f"Weight array count mismatch: model has {len(model_weights)}, received {len(weights)}")
        return False

    # Check if shapes match
    for i, (model_w, w) in enumerate(zip(model_weights, weights)):
        if model_w.shape != w.shape:
            logging.warning(f"Shape mismatch at layer {i}: model shape {model_w.shape}, received shape {w.shape}")
            return False

    return True

def adapt_weights(model, weights):
    """
    Adapt weights to be compatible with the model when possible.

    Args:
        model: Keras model
        weights: List of weight arrays

    Returns:
        Adapted weights if possible, None otherwise
    """
    if model is None:
        return None

    model_weights = model.get_weights()
    adapted_weights = []

    # Special case for LeNet model
    if len(model_weights) == 10 and len(weights) > 10:
        logging.info("Attempting to adapt weights for LeNet model")
        try:
            # For LeNet, we'll try to use just the first 10 weight arrays if they match in shape
            lenet_weights = weights[:10]

            # Check if the shapes match
            shapes_match = True
            for i, (model_w, w) in enumerate(zip(model_weights, lenet_weights)):
                if model_w.shape != w.shape:
                    logging.warning(f"LeNet adaptation: Shape mismatch at layer {i}: model shape {model_w.shape}, received shape {w.shape}")
                    shapes_match = False
                    break

            if shapes_match:
                logging.info("Successfully adapted weights for LeNet model")
                return lenet_weights

            # If shapes don't match, try to adapt each tensor
            adapted_lenet_weights = []
            for i, (model_w, w) in enumerate(zip(model_weights, lenet_weights)):
                if model_w.shape == w.shape:
                    adapted_lenet_weights.append(w)
                elif len(model_w.shape) == len(w.shape):
                    try:
                        adapted_w = adapt_tensor(w, model_w.shape)
                        adapted_lenet_weights.append(adapted_w)
                    except Exception as e:
                        logging.error(f"Error adapting tensor at layer {i}: {e}")
                        return None
                else:
                    logging.error(f"Cannot adapt tensor at layer {i}: different dimensions")
                    return None

            if len(adapted_lenet_weights) == len(model_weights):
                logging.info("Successfully adapted weights for LeNet model with tensor adaptation")
                return adapted_lenet_weights
        except Exception as e:
            logging.error(f"Error during LeNet weight adaptation: {e}")
            return None

    # Check if we can adapt the weights
    if len(model_weights) != len(weights):
        # Try to handle the case where the server model has fewer layers
        if len(weights) < len(model_weights):
            # Initialize with model weights
            adapted_weights = list(model_weights)
            # Replace matching layers with server weights
            for i, w in enumerate(weights):
                if i < len(model_weights) and model_weights[i].shape == w.shape:
                    adapted_weights[i] = w
            logging.info(f"Adapted weights by filling in missing layers: {len(adapted_weights)} layers")
            return adapted_weights
        logging.error(f"Cannot adapt weights: model has {len(model_weights)} layers, received {len(weights)} layers")
        return None

    # Try to adapt each weight array
    for i, (model_w, w) in enumerate(zip(model_weights, weights)):
        if model_w.shape == w.shape:
            # Shapes match, use as is
            adapted_weights.append(w)
        elif len(model_w.shape) == len(w.shape):
            # Same number of dimensions but different shape
            try:
                # Try to pad or truncate
                adapted_w = adapt_tensor(w, model_w.shape)
                adapted_weights.append(adapted_w)
            except Exception as e:
                logging.error(f"Error adapting tensor at layer {i}: {e}")
                return None
        else:
            # Different dimensions, cannot adapt
            logging.error(f"Cannot adapt tensor at layer {i}: different dimensions")
            return None

    logging.info(f"Successfully adapted all {len(adapted_weights)} weight arrays")
    return adapted_weights

def adapt_tensor(tensor, target_shape):
    """
    Adapt a tensor to a target shape by padding or truncating.

    Args:
        tensor: Source tensor
        target_shape: Target shape

    Returns:
        Adapted tensor
    """
    # Create a new tensor with the target shape, initialized with zeros
    adapted = np.zeros(target_shape, dtype=tensor.dtype)

    # Determine the slice sizes for each dimension
    slice_sizes = [min(s, t) for s, t in zip(tensor.shape, target_shape)]

    # Create slices for source and destination
    slices_src = tuple(slice(0, size) for size in slice_sizes)
    slices_dst = tuple(slice(0, size) for size in slice_sizes)

    # Copy the data from the source tensor to the adapted tensor
    adapted[slices_dst] = tensor[slices_src]

    return adapted
