"""
Dataset Generation Module for Federated Learning Simulation Platform

This module generates deterministic synthetic datasets for federated learning experiments.
The dataset simulates a binary classification problem with controllable complexity.

Key Features:
- Deterministic generation using fixed random seeds
- Non-IID client partitioning (simulates realistic federated scenarios)
- Disjoint client datasets (no data leakage)
- Validation and test set creation

Dataset Strategy:
- Generate synthetic features using make_classification
- Partition data across clients with statistical heterogeneity
- Each client receives private local training data
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from config import ExperimentConfig


def generate_synthetic_dataset(random_seed=None):
    """
    Generate a deterministic synthetic binary classification dataset.
    
    Args:
        random_seed: Random seed for reproducibility (uses config if None)
    
    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
    """
    if random_seed is None:
        random_seed = ExperimentConfig.RANDOM_SEED
    
    X, y = make_classification(
        n_samples=ExperimentConfig.DATASET_SIZE,
        n_features=ExperimentConfig.NUM_FEATURES,
        n_informative=15,
        n_redundant=3,
        n_repeated=0,
        n_classes=ExperimentConfig.NUM_CLASSES,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        hypercube=True,
        shift=0.0,
        scale=1.0,
        shuffle=True,
        random_state=random_seed
    )
    
    return X, y


def partition_data_for_clients(X, y, num_clients, random_seed=None):
    """
    Partition dataset into disjoint subsets for federated clients.
    
    Implements non-IID partitioning to simulate realistic heterogeneous client data.
    Each client receives data with potential class imbalance and different distributions.
    
    Args:
        X: Feature matrix
        y: Label vector
        num_clients: Number of clients to partition data for
        random_seed: Random seed for reproducibility
    
    Returns:
        client_datasets: List of tuples [(X_client_0, y_client_0), ..., (X_client_n, y_client_n)]
    """
    if random_seed is None:
        random_seed = ExperimentConfig.RANDOM_SEED

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if num_clients > len(y):
        raise ValueError(f"num_clients ({num_clients}) cannot exceed dataset size ({len(y)})")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    np.random.seed(random_seed)

    # Create class-based partitioning for non-IID simulation
    class_indices = {}
    for class_label in range(ExperimentConfig.NUM_CLASSES):
        class_indices[class_label] = np.where(y == class_label)[0]
        np.random.shuffle(class_indices[class_label])

    client_datasets = []
    alpha = 0.5  # Lower = more heterogeneous
    min_samples_per_client = 10
    base_samples = max(min_samples_per_client, len(y) // num_clients)

    for client_id in range(num_clients):
        client_X_list = []
        client_y_list = []

        proportions = np.random.dirichlet([alpha] * ExperimentConfig.NUM_CLASSES)

        for class_label in range(ExperimentConfig.NUM_CLASSES):
            n_samples_for_class = int(proportions[class_label] * base_samples)
            if n_samples_for_class == 0:
                continue

            available_indices = class_indices[class_label]
            if len(available_indices) == 0:
                continue

            start_idx = min((len(available_indices) // num_clients) * client_id, len(available_indices))
            end_idx = min(start_idx + n_samples_for_class, len(available_indices))
            selected_indices = available_indices[start_idx:end_idx]

            if len(selected_indices) > 0:
                client_X_list.append(X[selected_indices])
                client_y_list.append(y[selected_indices])

        if client_X_list:
            client_X = np.vstack(client_X_list)
            client_y = np.concatenate(client_y_list)

            shuffle_indices = np.random.permutation(len(client_y))
            client_X = client_X[shuffle_indices]
            client_y = client_y[shuffle_indices]
            client_datasets.append((client_X, client_y))
        else:
            fallback_indices = np.random.choice(len(y), size=min_samples_per_client, replace=False)
            client_datasets.append((X[fallback_indices], y[fallback_indices]))

    if len(client_datasets) != num_clients:
        raise RuntimeError(f"Failed to create {num_clients} client datasets, got {len(client_datasets)}")

    return client_datasets


def create_centralized_datasets(X, y, validation_split=None, test_split=0.15, random_seed=None):
    """
    Create train/validation/test splits for centralized baseline training.
    
    Args:
        X: Feature matrix
        y: Label vector
        validation_split: Proportion for validation set (uses config if None)
        test_split: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    if validation_split is None:
        validation_split = ExperimentConfig.VALIDATION_SPLIT
    
    if random_seed is None:
        random_seed = ExperimentConfig.RANDOM_SEED

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if not (0 < test_split < 1):
        raise ValueError("test_split must be in (0, 1)")
    if not (0 < validation_split < 1):
        raise ValueError("validation_split must be in (0, 1)")
    if test_split + validation_split >= 1:
        raise ValueError("test_split + validation_split must be < 1")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )
    
    # Second split: separate validation set from training
    adjusted_val_split = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_split,
        random_state=random_seed, stratify=y_temp
    )
    
    return {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test)
    }


def get_dataset_statistics(client_datasets):
    """
    Compute statistics about the federated dataset partitioning.
    
    Args:
        client_datasets: List of client datasets
    
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        "num_clients": len(client_datasets),
        "total_samples": sum(len(y) for _, y in client_datasets),
        "samples_per_client": [len(y) for _, y in client_datasets],
        "class_distribution_per_client": []
    }
    
    for client_id, (_, y) in enumerate(client_datasets):
        class_counts = {}
        for class_label in range(ExperimentConfig.NUM_CLASSES):
            class_counts[class_label] = np.sum(y == class_label)
        stats["class_distribution_per_client"].append(class_counts)
    
    return stats


if __name__ == "__main__":
    # Test dataset generation
    print("Generating synthetic dataset...")
    ExperimentConfig.set_random_seeds()
    
    X, y = generate_synthetic_dataset()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    print("\nPartitioning data for clients...")
    client_datasets = partition_data_for_clients(X, y, ExperimentConfig.NUM_CLIENTS)
    
    stats = get_dataset_statistics(client_datasets)
    print(f"\nDataset Statistics:")
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples per client: {stats['samples_per_client']}")
    
    print("\nClass distribution per client:")
    for i, dist in enumerate(stats['class_distribution_per_client']):
        print(f"  Client {i}: {dist}")
    
    print("\nCreating centralized datasets...")
    centralized_data = create_centralized_datasets(X, y)
    print(f"Train: {centralized_data['train'][0].shape}")
    print(f"Validation: {centralized_data['validation'][0].shape}")
    print(f"Test: {centralized_data['test'][0].shape}")
