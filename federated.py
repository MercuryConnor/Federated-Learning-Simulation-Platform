"""
Federated Learning Pipeline for Federated Learning Simulation Platform

This module implements the federated learning training approach using
TensorFlow Federated and the Federated Averaging (FedAvg) algorithm.

Federated Learning Strategy:
- Simulate multiple independent clients with private data
- Each round: select client subset, perform local training, aggregate updates
- Use FedAvg for model aggregation
- Track convergence across rounds
- Compare against centralized baseline

Implementation:
- TensorFlow Federated for simulation
- Client sampling with configurable participation rate
- Round-based training with comprehensive logging
- Model evaluation at each round

This represents the core federated learning experiment for evaluating
privacy-preserving distributed learning performance.
"""

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from datetime import datetime
import json
import os
from typing import List, Tuple

from config import ExperimentConfig
from model import create_model
from dataset import partition_data_for_clients


class FederatedTrainer:
    """Handles federated learning training simulation."""
    
    def __init__(self, client_datasets, test_data=None):
        """
        Initialize the federated trainer.
        
        Args:
            client_datasets: List of (X, y) tuples for each client
            test_data: Tuple of (X_test, y_test) for evaluation
        """
        self.client_datasets = client_datasets
        self.test_data = test_data
        self.num_clients = len(client_datasets)
        self.config = ExperimentConfig.get_config_summary()
        
        # Federated datasets in TFF format
        self.federated_train_data = None
        self.prepare_federated_datasets()
        
        # Training history
        self.round_metrics = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            "participating_clients": []
        }
        
        # Global model state
        self.global_model_weights = None
        
    def prepare_federated_datasets(self):
        """
        Convert client datasets to TensorFlow Federated format.
        
        Creates a list of tf.data.Dataset objects, one per client.
        """
        federated_data = []
        
        for client_id, (X, y) in enumerate(self.client_datasets):
            # Create TensorFlow dataset for this client
            client_dataset = tf.data.Dataset.from_tensor_slices({
                'x': X.astype(np.float32),
                'y': y.astype(np.float32).reshape(-1, 1)
            })
            
            # Batch and shuffle
            client_dataset = client_dataset.shuffle(len(y), seed=ExperimentConfig.RANDOM_SEED)
            client_dataset = client_dataset.batch(ExperimentConfig.BATCH_SIZE)
            
            federated_data.append(client_dataset)
        
        self.federated_train_data = federated_data
        print(f"Prepared {len(federated_data)} federated client datasets")
    
    def create_keras_model(self):
        """Create a Keras model for federated learning."""
        return create_model()
    
    def model_fn(self):
        """
        Create a TFF model wrapper for the Keras model.
        
        Returns:
            TFF model for federated learning
        """
        keras_model = self.create_keras_model()
        
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=self.federated_train_data[0].element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
    
    def train(self, num_rounds=None, client_fraction=None, local_epochs=None):
        """
        Train the federated model using Federated Averaging.
        
        Args:
            num_rounds: Number of federated learning rounds
            client_fraction: Fraction of clients to select per round
            local_epochs: Number of local epochs per client
        
        Returns:
            Training metrics history
        """
        # Use config defaults if not specified
        if num_rounds is None:
            num_rounds = ExperimentConfig.NUM_ROUNDS
        if client_fraction is None:
            client_fraction = ExperimentConfig.CLIENT_FRACTION
        if local_epochs is None:
            local_epochs = ExperimentConfig.LOCAL_EPOCHS
        
        print("="*70)
        print("FEDERATED LEARNING TRAINING")
        print("="*70)
        print(f"Number of clients: {self.num_clients}")
        print(f"Number of rounds: {num_rounds}")
        print(f"Client fraction per round: {client_fraction}")
        print(f"Local epochs: {local_epochs}")
        print(f"Batch size: {ExperimentConfig.BATCH_SIZE}")
        print("="*70)
        
        # Build federated averaging process
        iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=ExperimentConfig.LEARNING_RATE
            ),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                learning_rate=1.0  # Server learning rate
            )
        )
        
        # Initialize global model
        state = iterative_process.initialize()
        
        # Training loop
        for round_num in range(1, num_rounds + 1):
            # Select clients for this round
            num_selected_clients = max(1, int(self.num_clients * client_fraction))
            np.random.seed(ExperimentConfig.RANDOM_SEED + round_num)
            selected_indices = np.random.choice(
                self.num_clients, 
                size=num_selected_clients, 
                replace=False
            )
            selected_clients = [self.federated_train_data[i] for i in selected_indices]
            
            # Perform one round of federated training
            result = iterative_process.next(state, selected_clients)
            state = result.state
            round_metrics = result.metrics['client_work']['train']
            
            # Extract metrics
            train_loss = float(round_metrics['loss'])
            train_accuracy = float(round_metrics['binary_accuracy'])
            
            # Evaluate on test set if available
            test_loss, test_accuracy = self.evaluate_global_model(state)
            
            # Log metrics
            self.round_metrics["round"].append(round_num)
            self.round_metrics["train_loss"].append(train_loss)
            self.round_metrics["train_accuracy"].append(train_accuracy)
            self.round_metrics["test_loss"].append(test_loss)
            self.round_metrics["test_accuracy"].append(test_accuracy)
            self.round_metrics["participating_clients"].append(selected_indices.tolist())
            
            # Print progress
            if round_num % 5 == 0 or round_num == 1:
                print(f"Round {round_num:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"Test Acc: {test_accuracy:.4f} | "
                      f"Clients: {num_selected_clients}")
        
        # Save final global model weights
        self.global_model_weights = state
        
        print("="*70)
        print("FEDERATED TRAINING COMPLETE")
        print("="*70)
        
        return self.round_metrics
    
    def evaluate_global_model(self, state):
        """
        Evaluate the global model on test data.
        
        Args:
            state: TFF server state containing global model weights
        
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if self.test_data is None:
            return 0.0, 0.0
        
        X_test, y_test = self.test_data
        
        # Create a Keras model and set weights from state
        keras_model = self.create_keras_model()
        keras_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=ExperimentConfig.LEARNING_RATE),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )
        
        # Extract weights from TFF state
        model_weights = iterative_process_get_model_weights(state)
        keras_model.set_weights(model_weights)
        
        # Evaluate
        results = keras_model.evaluate(X_test, y_test, verbose=0)
        test_loss = float(results[0])
        test_accuracy = float(results[1])
        
        return test_loss, test_accuracy
    
    def get_final_test_metrics(self):
        """
        Get final test set metrics.
        
        Returns:
            Dictionary containing final test metrics
        """
        if len(self.round_metrics["round"]) == 0:
            raise ValueError("Model must be trained first")
        
        return {
            "test_loss": self.round_metrics["test_loss"][-1],
            "test_accuracy": self.round_metrics["test_accuracy"][-1],
            "final_round": self.round_metrics["round"][-1]
        }
    
    def save_results(self, save_dir=None):
        """
        Save federated training results to disk.
        
        Args:
            save_dir: Directory to save results
        
        Returns:
            Path to saved results file
        """
        if save_dir is None:
            save_dir = ExperimentConfig.RESULTS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Compile results
        results = {
            "experiment_type": "federated",
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "num_clients": self.num_clients,
            "round_metrics": self.round_metrics,
            "final_metrics": self.get_final_test_metrics()
        }
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"federated_results_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath


def iterative_process_get_model_weights(state):
    """
    Extract model weights from TFF server state.
    
    Args:
        state: TFF server state
    
    Returns:
        List of weight arrays
    """
    # This is a simplified extraction - actual implementation may vary
    # based on TFF version and model structure
    model_weights = []
    
    try:
        # Try to extract weights from state
        trainable_weights = state.model.trainable
        non_trainable_weights = state.model.non_trainable
        
        for weight in trainable_weights:
            model_weights.append(weight.numpy())
        for weight in non_trainable_weights:
            model_weights.append(weight.numpy())
    except:
        # Fallback: return empty if extraction fails
        pass
    
    return model_weights


def run_federated_experiment():
    """
    Run a complete federated learning experiment.
    
    This function orchestrates the entire federated training pipeline:
    1. Generate dataset
    2. Partition data for clients
    3. Create test set
    4. Train federated model
    5. Evaluate and save results
    
    Returns:
        Dictionary containing experiment results
    """
    print("="*70)
    print("FEDERATED LEARNING EXPERIMENT")
    print("="*70)
    
    # Set random seeds for reproducibility
    ExperimentConfig.set_random_seeds()
    
    # Generate dataset
    print("\n1. Generating synthetic dataset...")
    from dataset import generate_synthetic_dataset, create_centralized_datasets
    X, y = generate_synthetic_dataset()
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    
    # Partition for clients
    print("\n2. Partitioning data for federated clients...")
    client_datasets = partition_data_for_clients(
        X, y, 
        ExperimentConfig.NUM_CLIENTS
    )
    print(f"   Number of clients: {len(client_datasets)}")
    print(f"   Samples per client: {[len(y) for _, y in client_datasets]}")
    
    # Create test set (use centralized split for consistency)
    datasets = create_centralized_datasets(X, y)
    X_test, y_test = datasets['test']
    print(f"\n3. Test set: {X_test.shape}")
    
    # Train federated model
    print("\n4. Training federated model...")
    trainer = FederatedTrainer(client_datasets, test_data=(X_test, y_test))
    trainer.train()
    
    # Get final metrics
    print("\n5. Final test set metrics:")
    final_metrics = trainer.get_final_test_metrics()
    print(f"   - Loss: {final_metrics['test_loss']:.4f}")
    print(f"   - Accuracy: {final_metrics['test_accuracy']:.4f}")
    print(f"   - Final Round: {final_metrics['final_round']}")
    
    # Save results
    print("\n6. Saving results...")
    results_path = trainer.save_results()
    
    print("\n" + "="*70)
    print("FEDERATED EXPERIMENT COMPLETE")
    print("="*70)
    
    return {
        "trainer": trainer,
        "final_metrics": final_metrics,
        "results_path": results_path
    }


if __name__ == "__main__":
    # Run experiment
    results = run_federated_experiment()
