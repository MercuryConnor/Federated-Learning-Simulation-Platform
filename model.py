"""
Model Architecture Module for Federated Learning Simulation Platform

This module defines the shared neural network architecture used by both
centralized and federated training pipelines.

Design Principles:
- Simple, interpretable architecture
- Suitable for binary classification
- Compatible with TensorFlow Federated
- Parameterized for easy experimentation

The model must remain identical across both training paradigms to ensure
valid performance comparison.
"""

import numpy as np
import tensorflow as tf
from config import ExperimentConfig


def create_model():
    """
    Create a neural network model for binary classification.
    
    Architecture:
    - Input layer (size: NUM_FEATURES)
    - Dense hidden layers with ReLU activation
    - Dropout for regularization
    - Output layer with sigmoid activation (binary classification)
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(tf.keras.layers.Input(shape=(ExperimentConfig.NUM_FEATURES,)))
    
    # Hidden layers
    for units in ExperimentConfig.HIDDEN_UNITS:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(ExperimentConfig.DROPOUT_RATE))
    
    # Output layer for binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model


def compile_model(model, learning_rate=None):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer (uses config if None)
    
    Returns:
        Compiled model
    """
    if learning_rate is None:
        learning_rate = ExperimentConfig.LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_compiled_model(learning_rate=None):
    """
    Create and compile a model in one step.
    
    Args:
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model ready for training
    """
    model = create_model()
    model = compile_model(model, learning_rate)
    return model


def get_model_summary():
    """
    Get a string representation of the model architecture.
    
    Returns:
        String containing model summary
    """
    model = create_model()
    
    # Capture model summary as string
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    return "\n".join(summary_lines)


def count_model_parameters():
    """
    Count total and trainable parameters in the model.
    
    Returns:
        Dictionary with parameter counts
    """
    model = create_model()

    # Count parameters safely across TensorFlow versions
    try:
        trainable_params = sum(int(tf.size(w)) for w in model.trainable_weights)
        non_trainable_params = sum(int(tf.size(w)) for w in model.non_trainable_weights)
    except (AttributeError, TypeError):
        trainable_params = sum(np.prod(w.shape) for w in model.trainable_weights)
        non_trainable_params = sum(np.prod(w.shape) for w in model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params
    
    return {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "non_trainable": int(non_trainable_params)
    }


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    ExperimentConfig.set_random_seeds()
    
    model = create_compiled_model()
    
    print("\nModel Summary:")
    print(get_model_summary())
    
    params = count_model_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")
    
    # Test forward pass
    import numpy as np
    dummy_input = np.random.randn(10, ExperimentConfig.NUM_FEATURES).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    print(f"\nTest forward pass shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:3].flatten()}")
