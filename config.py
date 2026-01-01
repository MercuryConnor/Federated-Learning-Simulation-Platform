"""
Configuration Module for Federated Learning Simulation Platform

This module defines all global parameters for reproducible experimentation.
All experiments must use these configurations to ensure deterministic behavior.

Modify these parameters to conduct different experiments, but always maintain
fixed random seeds for reproducibility.
"""

import numpy as np
import tensorflow as tf


class ExperimentConfig:
    """Global configuration for federated learning experiments."""
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Dataset Parameters
    DATASET_SIZE = 10000  # Total number of samples
    NUM_FEATURES = 20     # Input feature dimension
    NUM_CLASSES = 2       # Binary classification
    
    # Federated Learning Parameters
    NUM_CLIENTS = 10              # Number of simulated clients
    NUM_ROUNDS = 50               # Federated learning rounds
    CLIENT_FRACTION = 0.3         # Fraction of clients per round (30%)
    LOCAL_EPOCHS = 5              # Local training epochs per client
    
    # Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Centralized Training Parameters
    CENTRALIZED_EPOCHS = 100  # Total epochs for centralized baseline
    VALIDATION_SPLIT = 0.2    # Validation set proportion
    
    # Model Architecture
    HIDDEN_UNITS = [64, 32]   # Hidden layer sizes
    DROPOUT_RATE = 0.3
    
    # Logging
    VERBOSE = 1
    LOG_DIR = "experiments/logs"
    RESULTS_DIR = "experiments/results"
    FIGURES_DIR = "experiments/figures"
    
    @staticmethod
    def validate_config():
        """Validate configuration parameters for correctness."""
        errors = []
        
        if not isinstance(ExperimentConfig.RANDOM_SEED, int):
            errors.append("RANDOM_SEED must be an integer")
        if not (0 < ExperimentConfig.CLIENT_FRACTION <= 1):
            errors.append("CLIENT_FRACTION must be in (0, 1]")
        if not (0 < ExperimentConfig.VALIDATION_SPLIT < 1):
            errors.append("VALIDATION_SPLIT must be in (0, 1)")
        if not (0 <= ExperimentConfig.DROPOUT_RATE < 1):
            errors.append("DROPOUT_RATE must be in [0, 1)")
        if ExperimentConfig.NUM_CLIENTS <= 0:
            errors.append("NUM_CLIENTS must be positive")
        if ExperimentConfig.NUM_ROUNDS <= 0:
            errors.append("NUM_ROUNDS must be positive")
        if ExperimentConfig.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        if ExperimentConfig.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        if ExperimentConfig.DATASET_SIZE <= 0:
            errors.append("DATASET_SIZE must be positive")
        if ExperimentConfig.NUM_FEATURES <= 0:
            errors.append("NUM_FEATURES must be positive")
        if ExperimentConfig.NUM_CLASSES <= 0:
            errors.append("NUM_CLASSES must be positive")
        if ExperimentConfig.DATASET_SIZE < ExperimentConfig.NUM_CLIENTS:
            errors.append("DATASET_SIZE must be >= NUM_CLIENTS")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @staticmethod
    def set_random_seeds():
        """Set all random seeds for reproducibility."""
        if not isinstance(ExperimentConfig.RANDOM_SEED, int):
            raise ValueError("RANDOM_SEED must be an integer")
        np.random.seed(ExperimentConfig.RANDOM_SEED)
        tf.random.set_seed(ExperimentConfig.RANDOM_SEED)
    
    @staticmethod
    def get_config_summary():
        """Return a dictionary of current configuration."""
        return {
            "random_seed": ExperimentConfig.RANDOM_SEED,
            "dataset_size": ExperimentConfig.DATASET_SIZE,
            "num_features": ExperimentConfig.NUM_FEATURES,
            "num_classes": ExperimentConfig.NUM_CLASSES,
            "num_clients": ExperimentConfig.NUM_CLIENTS,
            "num_rounds": ExperimentConfig.NUM_ROUNDS,
            "client_fraction": ExperimentConfig.CLIENT_FRACTION,
            "local_epochs": ExperimentConfig.LOCAL_EPOCHS,
            "batch_size": ExperimentConfig.BATCH_SIZE,
            "learning_rate": ExperimentConfig.LEARNING_RATE,
            "centralized_epochs": ExperimentConfig.CENTRALIZED_EPOCHS,
            "hidden_units": ExperimentConfig.HIDDEN_UNITS,
            "dropout_rate": ExperimentConfig.DROPOUT_RATE,
        }
