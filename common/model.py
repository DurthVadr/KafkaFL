"""
Model definition module for federated learning system.
Provides consistent model architecture across server and clients.
"""

import logging
import numpy as np

# Import TensorFlow conditionally
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available in model.py")
    TENSORFLOW_AVAILABLE = False

# Current model version
MODEL_VERSION = "v1"

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
        # Simple CNN for CIFAR-10
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

def get_random_weights():
    """
    Generate random weights for the model when TensorFlow is not available.
    
    Returns:
        List of numpy arrays with random weights
    """
    if TENSORFLOW_AVAILABLE:
        # Create model and get its weight shapes
        model = create_cifar10_model()
        if model is None:
            return [np.random.rand(10).astype(np.float32)]
        
        # Initialize with random weights
        weights = [np.random.normal(0, 0.05, w.shape).astype(np.float32) for w in model.get_weights()]
        return weights
    else:
        # Approximate shapes for the model defined in create_cifar10_model
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
    
    # Check if number of weight arrays matches
    if len(model_weights) != len(weights):
        return False
    
    # Check if shapes match
    for i, (model_w, w) in enumerate(zip(model_weights, weights)):
        if model_w.shape != w.shape:
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
            return adapted_weights
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
            except Exception:
                return None
        else:
            # Different dimensions, cannot adapt
            return None
    
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
    
    # For each dimension, determine the slice to copy
    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
    
    # Copy the data from the source tensor to the adapted tensor
    adapted[slices_dst] = tensor[slices_src]
    
    return adapted
