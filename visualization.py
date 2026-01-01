"""
Visualization Module for Federated Learning Simulation Platform

This module provides comprehensive visualization and comparison tools for
analyzing centralized vs federated learning performance.

Visualization Components:
- Training convergence curves
- Accuracy comparison charts
- Loss comparison charts
- Performance summary tables
- Client participation analysis

All visualizations must be:
- Interpretable for research analysis
- Publication-ready quality
- Reproducible and deterministic
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime

from config import ExperimentConfig


# Set publication-ready style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11


def plot_training_convergence(centralized_metrics, federated_metrics, save_path=None):
    """
    Plot training convergence comparison between centralized and federated.
    
    Args:
        centralized_metrics: Dictionary from centralized training
        federated_metrics: Dictionary from federated training
        save_path: Path to save figure (optional)
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract centralized metrics
    cent_epochs = range(1, len(centralized_metrics['loss']) + 1)
    cent_train_loss = centralized_metrics['loss']
    cent_val_loss = centralized_metrics['val_loss']
    cent_train_acc = centralized_metrics['accuracy']
    cent_val_acc = centralized_metrics['val_accuracy']
    
    # Extract federated metrics
    fed_rounds = federated_metrics['round']
    fed_train_loss = federated_metrics['train_loss']
    fed_test_loss = federated_metrics['test_loss']
    fed_train_acc = federated_metrics['train_accuracy']
    fed_test_acc = federated_metrics['test_accuracy']
    
    # Plot Loss
    axes[0].plot(cent_epochs, cent_train_loss, label='Centralized Train', 
                 linewidth=2, color='#2E86AB', linestyle='-')
    axes[0].plot(cent_epochs, cent_val_loss, label='Centralized Val', 
                 linewidth=2, color='#2E86AB', linestyle='--')
    axes[0].plot(fed_rounds, fed_train_loss, label='Federated Train', 
                 linewidth=2, color='#A23B72', linestyle='-', marker='o', markersize=4, markevery=5)
    axes[0].plot(fed_rounds, fed_test_loss, label='Federated Test', 
                 linewidth=2, color='#A23B72', linestyle='--', marker='s', markersize=4, markevery=5)
    
    axes[0].set_xlabel('Epoch / Round')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[1].plot(cent_epochs, cent_train_acc, label='Centralized Train', 
                 linewidth=2, color='#2E86AB', linestyle='-')
    axes[1].plot(cent_epochs, cent_val_acc, label='Centralized Val', 
                 linewidth=2, color='#2E86AB', linestyle='--')
    axes[1].plot(fed_rounds, fed_train_acc, label='Federated Train', 
                 linewidth=2, color='#A23B72', linestyle='-', marker='o', markersize=4, markevery=5)
    axes[1].plot(fed_rounds, fed_test_acc, label='Federated Test', 
                 linewidth=2, color='#A23B72', linestyle='--', marker='s', markersize=4, markevery=5)
    
    axes[1].set_xlabel('Epoch / Round')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    return fig


def plot_final_comparison(centralized_results, federated_results, save_path=None):
    """
    Plot bar chart comparing final test set performance.
    
    Args:
        centralized_results: Dictionary with centralized test metrics
        federated_results: Dictionary with federated test metrics
        save_path: Path to save figure (optional)
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract metrics
    cent_acc = centralized_results.get('test_accuracy', 0)
    cent_loss = centralized_results.get('test_loss', 0)
    fed_acc = federated_results.get('test_accuracy', 0)
    fed_loss = federated_results.get('test_loss', 0)
    
    # Accuracy comparison
    methods = ['Centralized', 'Federated']
    accuracies = [cent_acc, fed_acc]
    colors = ['#2E86AB', '#A23B72']
    
    bars1 = axes[0].bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Set Accuracy Comparison')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Loss comparison
    losses = [cent_loss, fed_loss]
    bars2 = axes[1].bar(methods, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Test Set Loss Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_client_participation(federated_metrics, save_path=None):
    """
    Visualize client participation patterns across rounds.
    
    Args:
        federated_metrics: Dictionary with federated training metrics
        save_path: Path to save figure (optional)
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rounds = federated_metrics['round']
    participating_clients = federated_metrics['participating_clients']
    
    # Create participation matrix
    num_rounds = len(rounds)
    num_clients = max(max(clients) for clients in participating_clients) + 1
    
    participation_matrix = np.zeros((num_clients, num_rounds))
    
    for round_idx, clients in enumerate(participating_clients):
        for client_id in clients:
            participation_matrix[client_id, round_idx] = 1
    
    # Plot heatmap
    sns.heatmap(participation_matrix, cmap='YlOrRd', cbar_kws={'label': 'Participated'},
                xticklabels=rounds[::5], yticklabels=range(num_clients),
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Client ID')
    ax.set_title('Client Participation Heatmap Across Federated Rounds')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Client participation plot saved to: {save_path}")
    
    return fig


def create_performance_summary_table(centralized_results, federated_results):
    """
    Create a formatted summary table of performance metrics.
    
    Args:
        centralized_results: Dictionary with centralized metrics
        federated_results: Dictionary with federated metrics
    
    Returns:
        String containing formatted table
    """
    table = []
    table.append("="*70)
    table.append("PERFORMANCE SUMMARY: CENTRALIZED VS FEDERATED")
    table.append("="*70)
    table.append("")
    table.append(f"{'Metric':<25} {'Centralized':>15} {'Federated':>15} {'Difference':>15}")
    table.append("-"*70)
    
    # Accuracy
    cent_acc = centralized_results.get('test_accuracy', 0)
    fed_acc = federated_results.get('test_accuracy', 0)
    diff_acc = cent_acc - fed_acc
    table.append(f"{'Test Accuracy':<25} {cent_acc:>15.4f} {fed_acc:>15.4f} {diff_acc:>+15.4f}")
    
    # Loss
    cent_loss = centralized_results.get('test_loss', 0)
    fed_loss = federated_results.get('test_loss', 0)
    diff_loss = cent_loss - fed_loss
    table.append(f"{'Test Loss':<25} {cent_loss:>15.4f} {fed_loss:>15.4f} {diff_loss:>+15.4f}")
    
    # Additional metrics if available
    if 'test_precision' in centralized_results:
        cent_prec = centralized_results['test_precision']
        table.append(f"{'Precision':<25} {cent_prec:>15.4f} {'N/A':>15} {'N/A':>15}")
    
    if 'test_recall' in centralized_results:
        cent_rec = centralized_results['test_recall']
        table.append(f"{'Recall':<25} {cent_rec:>15.4f} {'N/A':>15} {'N/A':>15}")
    
    if 'test_auc' in centralized_results:
        cent_auc = centralized_results['test_auc']
        table.append(f"{'AUC':<25} {cent_auc:>15.4f} {'N/A':>15} {'N/A':>15}")
    
    table.append("="*70)
    table.append("")
    table.append("INTERPRETATION:")
    table.append(f"  - Accuracy Gap: {abs(diff_acc)*100:.2f}% {'(Centralized better)' if diff_acc > 0 else '(Federated better)'}")
    table.append(f"  - Loss Gap: {abs(diff_loss):.4f} {'(Centralized better)' if diff_loss < 0 else '(Federated better)'}")
    table.append("")
    table.append("="*70)
    
    return "\n".join(table)


def generate_all_visualizations(centralized_path, federated_path, output_dir=None):
    """
    Generate all visualizations from saved experiment results.
    
    Args:
        centralized_path: Path to centralized results JSON
        federated_path: Path to federated results JSON
        output_dir: Directory to save figures (uses config if None)
    
    Returns:
        Dictionary of generated figure paths
    """
    if output_dir is None:
        output_dir = ExperimentConfig.FIGURES_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(centralized_path, 'r') as f:
        centralized_data = json.load(f)
    
    with open(federated_path, 'r') as f:
        federated_data = json.load(f)
    
    # Extract metrics
    cent_metrics = centralized_data['training_metrics']
    cent_test = centralized_data['test_metrics']
    fed_metrics = federated_data['round_metrics']
    fed_test = federated_data['final_metrics']
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate plots
    figures = {}
    
    # 1. Convergence plot
    conv_path = os.path.join(output_dir, f"convergence_{timestamp}.png")
    plot_training_convergence(cent_metrics, fed_metrics, save_path=conv_path)
    figures['convergence'] = conv_path
    
    # 2. Final comparison plot
    comp_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
    plot_final_comparison(cent_test, fed_test, save_path=comp_path)
    figures['comparison'] = comp_path
    
    # 3. Client participation plot
    part_path = os.path.join(output_dir, f"client_participation_{timestamp}.png")
    plot_client_participation(fed_metrics, save_path=part_path)
    figures['participation'] = part_path
    
    # 4. Generate summary table
    summary = create_performance_summary_table(cent_test, fed_test)
    print("\n" + summary)
    
    # Save summary to file
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    figures['summary'] = summary_path
    
    print(f"\nAll visualizations generated in: {output_dir}")
    
    return figures


def _latest_result_file(results_dir, prefix):
    """Return the newest result file that matches the given prefix."""
    candidates = [
        f for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith('.json')
    ]
    if not candidates:
        raise FileNotFoundError(f"No result files found with prefix '{prefix}' in {results_dir}")
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    return os.path.join(results_dir, candidates[0])


def generate_comparison_plots():
    """Generate accuracy and loss comparison charts using the latest results."""
    results_dir = ExperimentConfig.RESULTS_DIR
    figures_dir = ExperimentConfig.FIGURES_DIR

    os.makedirs(figures_dir, exist_ok=True)

    # Locate most recent centralized and federated results
    cent_path = _latest_result_file(results_dir, 'centralized_results_')
    fed_path = _latest_result_file(results_dir, 'federated_results_')

    with open(cent_path, 'r') as f:
        centralized_data = json.load(f)
    with open(fed_path, 'r') as f:
        federated_data = json.load(f)

    cent_metrics = centralized_data['training_metrics']
    fed_metrics = federated_data['round_metrics']

    # Accuracy plot
    acc_fig, acc_ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(cent_metrics['accuracy']) + 1)
    acc_ax.plot(epochs, cent_metrics['accuracy'], label='Centralized Train', color='#2E86AB', linewidth=2)
    if 'val_accuracy' in cent_metrics:
        acc_ax.plot(epochs, cent_metrics['val_accuracy'], label='Centralized Val', color='#2E86AB', linestyle='--', linewidth=2)
    rounds = fed_metrics['round']
    acc_ax.plot(rounds, fed_metrics['train_accuracy'], label='Federated Train', color='#A23B72', linewidth=2, marker='o', markersize=4, markevery=5)
    acc_ax.plot(rounds, fed_metrics['test_accuracy'], label='Federated Test', color='#A23B72', linestyle='--', linewidth=2, marker='s', markersize=4, markevery=5)
    acc_ax.set_xlabel('Epoch / Round')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_title('Federated vs Centralized Accuracy')
    acc_ax.grid(True, alpha=0.3)
    acc_ax.legend()
    acc_path = os.path.join(figures_dir, 'federated_vs_centralized_accuracy.png')
    acc_fig.tight_layout()
    acc_fig.savefig(acc_path, dpi=300, bbox_inches='tight')

    # Loss plot
    loss_fig, loss_ax = plt.subplots(figsize=(10, 5))
    loss_ax.plot(epochs, cent_metrics['loss'], label='Centralized Train', color='#2E86AB', linewidth=2)
    if 'val_loss' in cent_metrics:
        loss_ax.plot(epochs, cent_metrics['val_loss'], label='Centralized Val', color='#2E86AB', linestyle='--', linewidth=2)
    loss_ax.plot(rounds, fed_metrics['train_loss'], label='Federated Train', color='#A23B72', linewidth=2, marker='o', markersize=4, markevery=5)
    loss_ax.plot(rounds, fed_metrics['test_loss'], label='Federated Test', color='#A23B72', linestyle='--', linewidth=2, marker='s', markersize=4, markevery=5)
    loss_ax.set_xlabel('Epoch / Round')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Federated vs Centralized Loss')
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()
    loss_path = os.path.join(figures_dir, 'federated_vs_centralized_loss.png')
    loss_fig.tight_layout()
    loss_fig.savefig(loss_path, dpi=300, bbox_inches='tight')

    print("Generated comparison plots:")
    print(f"  Accuracy: {acc_path}")
    print(f"  Loss: {loss_path}")

    return {'accuracy': acc_path, 'loss': loss_path}


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully")
    print("Use generate_all_visualizations() with experiment results to create plots")
