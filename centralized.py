"""
Centralized Training Pipeline for Federated Learning Simulation Platform

This module implements the centralized baseline training approach.
The centralized model serves as the upper-bound performance reference for
comparing against federated learning results.

Training Strategy:
- Train on complete dataset (all client data combined)
- Use standard supervised learning with validation
- Log comprehensive metrics for comparison
- Serve as the gold standard for federated evaluation

This baseline is essential for understanding the performance trade-offs
inherent in federated learning.
"""

from datetime import datetime
import json
import os
import numpy as np

from config import ExperimentConfig
from model import create_compiled_model
from dataset import create_centralized_datasets


class CentralizedTrainer:
    """Handles centralized baseline model training."""
    
    def __init__(self):
        """Initialize the centralized trainer."""
        self.model = None
        self.history = None
        self.config = ExperimentConfig.get_config_summary()
        
    def train(self, X, y, epochs=None, batch_size=None, validation_split=None, verbose=None):
        """
        Train the centralized model.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs (uses config if None)
            batch_size: Batch size (uses config if None)
            validation_split: Validation split ratio (uses config if None)
            verbose: Verbosity level (uses config if None)
        
        Returns:
            Training history
        """
        # Use config defaults if not specified
        if epochs is None:
            epochs = ExperimentConfig.CENTRALIZED_EPOCHS
        if batch_size is None:
            batch_size = ExperimentConfig.BATCH_SIZE
        if validation_split is None:
            validation_split = ExperimentConfig.VALIDATION_SPLIT
        if verbose is None:
            verbose = ExperimentConfig.VERBOSE
        
        # Create and compile model
        ExperimentConfig.set_random_seeds()
        self.model = create_compiled_model()
        
        print(f"Training centralized model for {epochs} epochs...")
        print(f"Training samples: {len(X)}")
        print(f"Validation split: {validation_split}")
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        results = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)

        metrics = {
            "test_loss": float(results.get('loss', 0.0)),
            "test_accuracy": float(results.get('accuracy', 0.0)),
            "test_precision": float(results.get('precision', 0.0)),
            "test_recall": float(results.get('recall', 0.0)),
            "test_auc": float(results.get('auc', 0.0))
        }

        return metrics
    
    def get_training_metrics(self):
        """
        Extract training metrics from history.
        
        Returns:
            Dictionary containing training and validation metrics per epoch
        """
        if self.history is None:
            raise ValueError("Model must be trained first")
        
        return {
            "loss": self.history.history['loss'],
            "accuracy": self.history.history['accuracy'],
            "val_loss": self.history.history['val_loss'],
            "val_accuracy": self.history.history['val_accuracy'],
            "precision": self.history.history.get('precision', []),
            "recall": self.history.history.get('recall', []),
            "auc": self.history.history.get('auc', []),
        }
    
    def save_results(self, test_metrics, save_dir=None):
        """
        Save training results and metrics to disk.
        
        Args:
            test_metrics: Dictionary of test set metrics
            save_dir: Directory to save results (uses config if None)
        
        Returns:
            Path to saved results file
        """
        if save_dir is None:
            save_dir = ExperimentConfig.RESULTS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Compile all results
        results = {
            "experiment_type": "centralized",
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "training_metrics": self.get_training_metrics(),
            "test_metrics": test_metrics,
            "final_epoch_metrics": {
                "train_loss": float(self.history.history['loss'][-1]),
                "train_accuracy": float(self.history.history['accuracy'][-1]),
                "val_loss": float(self.history.history['val_loss'][-1]),
                "val_accuracy": float(self.history.history['val_accuracy'][-1]),
            }
        }
        
        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"centralized_results_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def save_model(self, save_dir=None):
        """
        Save the trained model to disk.
        
        Args:
            save_dir: Directory to save model (uses config if None)
        
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        if save_dir is None:
            save_dir = ExperimentConfig.RESULTS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"centralized_model_{timestamp}.keras")
        
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")
        return model_path


def run_centralized_experiment():
    """
    Run a complete centralized training experiment.
    
    This function orchestrates the entire centralized training pipeline:
    1. Generate dataset
    2. Create train/val/test splits
    3. Train model
    4. Evaluate on test set
    5. Save results
    
    Returns:
        Dictionary containing experiment results
    """
    print("="*70)
    print("CENTRALIZED BASELINE EXPERIMENT")
    print("="*70)
    
    # Set random seeds for reproducibility
    ExperimentConfig.set_random_seeds()
    
    # Generate dataset
    print("\n1. Generating synthetic dataset...")
    from dataset import generate_synthetic_dataset
    X, y = generate_synthetic_dataset()
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    
    # Create splits
    print("\n2. Creating train/validation/test splits...")
    datasets = create_centralized_datasets(X, y)
    X_train, y_train = datasets['train']
    X_val, y_val = datasets['validation']
    X_test, y_test = datasets['test']
    
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")
    
    # Combine train and validation for centralized training
    # (model will internally split for validation during training)
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])
    
    # Train model
    print("\n3. Training centralized model...")
    trainer = CentralizedTrainer()
    trainer.train(X_full_train, y_full_train)
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    
    print("\n   Test Set Results:")
    print(f"   - Loss: {test_metrics['test_loss']:.4f}")
    print(f"   - Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"   - Precision: {test_metrics['test_precision']:.4f}")
    print(f"   - Recall: {test_metrics['test_recall']:.4f}")
    print(f"   - AUC: {test_metrics['test_auc']:.4f}")
    
    # Save results
    print("\n5. Saving results...")
    results_path = trainer.save_results(test_metrics)
    model_path = trainer.save_model()
    
    print("\n" + "="*70)
    print("CENTRALIZED EXPERIMENT COMPLETE")
    print("="*70)
    
    return {
        "trainer": trainer,
        "test_metrics": test_metrics,
        "results_path": results_path,
        "model_path": model_path
    }


if __name__ == "__main__":
    # Run experiment
    results = run_centralized_experiment()
