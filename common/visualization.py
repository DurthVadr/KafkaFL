"""
Visualization module for federated learning system.

This module provides functions to visualize and save metrics from the federated learning process.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import logging
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform

# Configure matplotlib for non-interactive backend
plt.switch_backend('agg')

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

def generate_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_plot(fig, filename, logger=None):
    """
    Save a matplotlib figure to the plots directory.

    Args:
        fig: Matplotlib figure to save
        filename: Base filename (without directory)
        logger: Optional logger for logging messages

    Returns:
        Path to the saved file
    """
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # Add timestamp to filename to avoid overwriting
    timestamp = generate_timestamp()
    full_filename = f"plots/{timestamp}_{filename}"

    # Save the figure
    fig.savefig(full_filename, dpi=300, bbox_inches='tight')

    if logger:
        logger.info(f"Plot saved to {full_filename}")
    else:
        print(f"Plot saved to {full_filename}")

    plt.close(fig)
    return full_filename

def plot_client_accuracy(accuracies, client_id, logger=None):
    """
    Plot training and test accuracy for a client.

    Args:
        accuracies: Dictionary with 'train' and 'test' lists of accuracy values
        client_id: ID of the client
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training accuracy if available
    if 'train' in accuracies and accuracies['train']:
        ax.plot(accuracies['train'], 'b-', label='Training Accuracy')

    # Plot test accuracy if available
    if 'test' in accuracies and accuracies['test']:
        ax.plot(accuracies['test'], 'r-', label='Test Accuracy')

    ax.set_xlabel('Training Cycle')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Client {client_id} Accuracy Over Time')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits for accuracy (0-1)
    ax.set_ylim(0, 1)

    return save_plot(fig, f"client_{client_id}_accuracy.png", logger)

def plot_client_loss(losses, client_id, logger=None):
    """
    Plot training loss for a client.

    Args:
        losses: List of loss values
        client_id: ID of the client
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses, 'g-')
    ax.set_xlabel('Training Cycle')
    ax.set_ylabel('Loss')
    ax.set_title(f'Client {client_id} Training Loss Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)

    return save_plot(fig, f"client_{client_id}_loss.png", logger)

def plot_server_aggregations(aggregation_times, update_counts, logger=None):
    """
    Plot server aggregation metrics.

    Args:
        aggregation_times: List of timestamps when aggregations occurred
        update_counts: List of number of updates received for each aggregation
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Convert timestamps to minutes from start
    if aggregation_times:
        start_time = aggregation_times[0]
        relative_times = [(t - start_time) / 60 for t in aggregation_times]

        # Plot cumulative aggregations
        ax1.plot(relative_times, range(1, len(aggregation_times) + 1), 'b-', label='Cumulative Aggregations')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Cumulative Aggregations', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot updates per aggregation on secondary y-axis
        if update_counts:
            ax2 = ax1.twinx()
            ax2.plot(relative_times, update_counts, 'r-', label='Updates per Aggregation')
            ax2.set_ylabel('Updates per Aggregation', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if update_counts:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')

        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Server Aggregation Metrics Over Time')

        return save_plot(fig, "server_aggregations.png", logger)

    return None

def plot_client_comparison(client_metrics, metric_name='test_accuracy', logger=None):
    """
    Plot comparison of metrics across different clients.

    Args:
        client_metrics: Dictionary mapping client IDs to lists of metric values
        metric_name: Name of the metric being compared (for title and filename)
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for client_id, metrics in client_metrics.items():
        if metrics:  # Only plot if there are metrics
            ax.plot(metrics, label=f'Client {client_id}')

    ax.set_xlabel('Training Cycle')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison Across Clients')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits for accuracy metrics (0-1)
    if 'accuracy' in metric_name.lower():
        ax.set_ylim(0, 1)

    return save_plot(fig, f"client_comparison_{metric_name}.png", logger)

def plot_global_performance(rounds, accuracies, logger=None):
    """
    Plot global model performance over rounds.

    Args:
        rounds: List of round numbers
        accuracies: List of accuracy values for the global model
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds, accuracies, 'b-o')
    ax.set_xlabel('Aggregation Round')
    ax.set_ylabel('Global Model Accuracy')
    ax.set_title('Global Model Performance Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set y-axis limits for accuracy (0-1)
    ax.set_ylim(0, 1)

    return save_plot(fig, "global_model_performance.png", logger)

def plot_weight_distribution_violin(weights, round_num, layer_indices=None, logger=None):
    """
    Create violin plots showing the distribution of weights in selected layers.

    Args:
        weights: List of weight arrays from the model
        round_num: Aggregation round number for the title
        layer_indices: List of layer indices to plot (default: first 4 weight layers)
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    # If no layer indices provided, use the first few weight layers (skip bias layers)
    if layer_indices is None:
        # Find weight layers (typically these are even-indexed in the weights list)
        layer_indices = [i for i in range(len(weights)) if len(weights[i].shape) > 1][:4]

    # Create figure with subplots for each selected layer
    fig, axes = plt.subplots(len(layer_indices), 1, figsize=(10, 3*len(layer_indices)))

    # Handle case with only one layer
    if len(layer_indices) == 1:
        axes = [axes]

    for i, layer_idx in enumerate(layer_indices):
        if layer_idx >= len(weights):
            if logger:
                logger.warning(f"Layer index {layer_idx} out of range. Skipping.")
            continue

        # Flatten the weights for the violin plot
        layer_weights = weights[layer_idx].flatten()

        # Create violin plot
        sns.violinplot(y=layer_weights, ax=axes[i], inner="quartile", color="skyblue")

        # Add layer information
        axes[i].set_title(f"Layer {layer_idx} (shape: {weights[layer_idx].shape})")
        axes[i].set_ylabel("Weight Value")

        # Add statistics
        mean_val = np.mean(layer_weights)
        std_val = np.std(layer_weights)
        min_val = np.min(layer_weights)
        max_val = np.max(layer_weights)

        stats_text = f"Mean: {mean_val:.4f}, Std: {std_val:.4f}\nMin: {min_val:.4f}, Max: {max_val:.4f}"
        axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.suptitle(f"Weight Distribution at Round {round_num}", fontsize=16, y=1.02)

    return save_plot(fig, f"weight_distribution_round_{round_num}.png", logger)

def plot_convergence_visualization(weight_history, layer_idx=0, logger=None):
    """
    Visualize the convergence of weights over multiple rounds.

    Args:
        weight_history: List of weight arrays from different rounds
        layer_idx: Index of the layer to visualize
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    if not weight_history:
        if logger:
            logger.warning("No weight history available for convergence visualization")
        return None

    # Get the number of rounds
    num_rounds = len(weight_history)

    if num_rounds < 2:
        if logger:
            logger.warning("Need at least 2 rounds for convergence visualization")
        return None

    # Check if the specified layer exists in all rounds
    if any(layer_idx >= len(weights) for weights in weight_history):
        if logger:
            logger.error(f"Layer index {layer_idx} out of range for some rounds")
        return None

    # Calculate weight changes between consecutive rounds
    weight_changes = []
    for i in range(1, num_rounds):
        # Calculate the absolute difference between consecutive rounds
        change = np.abs(weight_history[i][layer_idx] - weight_history[i-1][layer_idx])
        weight_changes.append(change)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Heatmap of weight changes over time
    if len(weight_changes) > 1:
        # Reshape the weight changes for visualization if needed
        if len(weight_changes[0].shape) > 2:
            # For convolutional layers, take the mean across some dimensions
            reshaped_changes = [np.mean(change, axis=tuple(range(2, len(change.shape)))) for change in weight_changes]
        else:
            reshaped_changes = weight_changes

        # Create a heatmap-like visualization
        im = axes[0].imshow(np.array(reshaped_changes), aspect='auto', cmap='viridis')
        axes[0].set_xlabel('Weight Index')
        axes[0].set_ylabel('Round')
        axes[0].set_title(f'Weight Changes Over Time (Layer {layer_idx})')
        fig.colorbar(im, ax=axes[0], label='Absolute Change')
    else:
        axes[0].text(0.5, 0.5, "Not enough rounds for heatmap",
                    horizontalalignment='center', verticalalignment='center')

    # Plot 2: Line plot of average weight change per round
    avg_changes = [np.mean(change) for change in weight_changes]
    axes[1].plot(range(1, num_rounds), avg_changes, 'b-o')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Average Weight Change')
    axes[1].set_title('Convergence Rate')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    return save_plot(fig, f"convergence_visualization_layer_{layer_idx}.png", logger)

def plot_client_similarity_heatmap(client_updates, client_ids=None, logger=None):
    """
    Create a heatmap showing similarity between client model updates.

    Args:
        client_updates: Dictionary mapping client IDs to their model updates (weight arrays)
        client_ids: List of client IDs to include (default: all clients)
        logger: Optional logger for logging messages

    Returns:
        Path to the saved plot
    """
    if not client_updates:
        if logger:
            logger.warning("No client updates available for similarity heatmap")
        return None

    # If no client IDs provided, use all clients
    if client_ids is None:
        client_ids = list(client_updates.keys())

    # Filter client updates to include only specified clients
    client_updates = {cid: updates for cid, updates in client_updates.items() if cid in client_ids}

    if len(client_updates) < 2:
        if logger:
            logger.warning("Need at least 2 clients for similarity heatmap")
        return None

    # Flatten weights for each client to compute similarity
    flattened_weights = {}
    for client_id, weights in client_updates.items():
        # Concatenate all weight arrays into a single vector
        flattened = np.concatenate([w.flatten() for w in weights])
        flattened_weights[client_id] = flattened

    # Compute pairwise cosine similarity
    client_ids_list = list(flattened_weights.keys())
    n_clients = len(client_ids_list)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(n_clients):
            # Compute cosine similarity between client i and client j
            vec1 = flattened_weights[client_ids_list[i]]
            vec2 = flattened_weights[client_ids_list[j]]

            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)

            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            similarity_matrix[i, j] = similarity

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')

    # Set ticks and labels
    ax.set_xticks(np.arange(n_clients))
    ax.set_yticks(np.arange(n_clients))
    ax.set_xticklabels([f'Client {cid}' for cid in client_ids_list])
    ax.set_yticklabels([f'Client {cid}' for cid in client_ids_list])

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values to cells
    for i in range(n_clients):
        for j in range(n_clients):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                   ha="center", va="center", color="black" if abs(similarity_matrix[i, j]) < 0.7 else "white")

    ax.set_title("Client Similarity Heatmap (Cosine Similarity)")
    fig.tight_layout()

    return save_plot(fig, "client_similarity_heatmap.png", logger)
