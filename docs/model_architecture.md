# Model Architecture

This document describes the model architecture used in the federated learning system for the CIFAR-10 dataset.

## Overview

The model is a Convolutional Neural Network (CNN) designed for image classification on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.

## Architecture Details

The model follows a standard CNN architecture with the following layers:

```
Input: (32, 32, 3) - RGB images of size 32x32
↓
Conv2D: 16 filters, 3x3 kernel, 'valid' padding, ReLU activation
Output shape: (30, 30, 16)
↓
Conv2D: 32 filters, 3x3 kernel, 'valid' padding, ReLU activation
Output shape: (28, 28, 32)
↓
MaxPooling2D: 2x2 pool size
Output shape: (14, 14, 32)
↓
Dropout: 0.25 rate
Output shape: (14, 14, 32)
↓
Flatten
Output shape: (6272,)
↓
Dense: 64 units, ReLU activation
Output shape: (64,)
↓
Dropout: 0.5 rate
Output shape: (64,)
↓
Dense: 10 units, softmax activation (output layer)
Output shape: (10,)
```

## Implementation

### TensorFlow/Keras Implementation

```python
def create_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    inputs = Input(shape=input_shape)
    
    # First convolutional layer with explicit parameters
    x = Conv2D(16, (3, 3), padding='valid', activation='relu')(inputs)  # 30x30x16
    
    # Second convolutional layer
    x = Conv2D(32, (3, 3), padding='valid', activation='relu')(x)  # 28x28x32
    
    # Max pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)  # 14x14x32
    
    # Dropout for regularization
    x = Dropout(0.25)(x)
    
    # Flatten layer - should be 14*14*32 = 6272
    x = Flatten()(x)
    
    # Dense layer
    x = Dense(64, activation='relu')(x)
    
    # Dropout for regularization
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

## Model Weights

The model has the following weight arrays:

1. Conv2D weights: (3, 3, 3, 16) - First convolutional layer weights
2. Conv2D bias: (16,) - First convolutional layer bias
3. Conv2D weights: (3, 3, 16, 32) - Second convolutional layer weights
4. Conv2D bias: (32,) - Second convolutional layer bias
5. Dense weights: (6272, 64) - First dense layer weights after flattening
6. Dense bias: (64,) - First dense layer bias
7. Dense weights: (64, 10) - Output layer weights
8. Dense bias: (10,) - Output layer bias

## Design Considerations

1. **Simplicity**: The model is intentionally kept simple to reduce communication overhead in the federated learning setting.

2. **Explicit Padding**: We use 'valid' padding (no padding) to ensure consistent dimensions across different implementations.

3. **Regularization**: Dropout layers are included to prevent overfitting, which is especially important in federated learning where clients may have limited data.

4. **Compatibility**: The architecture is designed to be easily implementable across different platforms and devices.

## Performance

On the CIFAR-10 dataset, this model typically achieves:
- Training accuracy: ~70-80% after a few epochs
- Test accuracy: ~65-75%

While more complex architectures could achieve higher accuracy, this model provides a good balance between performance and communication efficiency for federated learning.

## Adaptation for Federated Learning

For federated learning, we've made the following adaptations:

1. **Reduced Size**: Fewer filters and units compared to state-of-the-art models to reduce communication overhead.

2. **Consistent Architecture**: Explicit parameters to ensure consistency between server and client implementations.

3. **Weight Adaptation**: The system includes mechanisms to handle minor differences in layer dimensions (see [Weight Adaptation](weight_adaptation.md)).

## Future Improvements

1. **Model Compression**: Implement techniques like pruning or quantization to further reduce communication overhead.

2. **Architecture Search**: Explore different architectures that balance performance and communication efficiency.

3. **Transfer Learning**: Incorporate pre-trained models to improve performance, especially for clients with limited data.
