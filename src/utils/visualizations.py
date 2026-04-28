"""
Visualization Utilities
Creates plots and visualizations for model evaluation
Week 6 - Visualization (Supporting: Aditi, Anuj, Jensi)
"""

import matplotlib
matplotlib.use('Agg')  # save to PNG files only -- no GUI windows, no CPU overhead
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import json


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix as heatmap
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix_normalized(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    """
    Plot normalized confusion matrix (percentage)
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='YlOrRd',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to {save_path}")
    
    plt.close()


def plot_per_class_accuracy(
    metrics: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Plot per-class accuracy as bar chart
    
    Args:
        metrics: Metrics dictionary from evaluator
        save_path: Path to save figure
        figsize: Figure size
    """
    # Extract data
    class_names = list(metrics['per_class_metrics'].keys())
    accuracies = [metrics['per_class_metrics'][c]['accuracy'] * 100 
                  for c in class_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(class_names, accuracies, color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Add overall accuracy line
    overall_acc = metrics['overall_accuracy'] * 100
    ax.axhline(y=overall_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Acc: {overall_acc:.2f}%')
    
    ax.set_xlabel('Location Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class accuracy plot saved to {save_path}")
    
    plt.close()


def plot_metrics_comparison(
    metrics: Dict,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6)
):
    """
    Plot comparison of precision, recall, F1 for each class
    
    Args:
        metrics: Metrics dictionary from evaluator
        save_path: Path to save figure
        figsize: Figure size
    """
    # Extract data
    class_names = list(metrics['per_class_metrics'].keys())
    precision = [metrics['per_class_metrics'][c]['precision'] * 100 for c in class_names]
    recall = [metrics['per_class_metrics'][c]['recall'] * 100 for c in class_names]
    f1_scores = [metrics['per_class_metrics'][c]['f1_score'] * 100 for c in class_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='lightcoral')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightskyblue')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Location Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1-Score by Class', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training history from JSON file
    
    Args:
        history_path: Path to training_history.json
        save_path: Path to save figure
        figsize: Figure size
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def create_all_visualizations(
    metrics: Dict,
    history_path: str,
    output_dir: str = 'results/visualizations'
):
    """
    Create all visualizations for evaluation
    
    Args:
        metrics: Evaluation metrics dictionary
        history_path: Path to training history JSON
        output_dir: Directory to save all plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Creating All Visualizations")
    print("="*60)
    
    # Confusion matrix
    print("\n1. Creating confusion matrix...")
    cm = np.array(metrics['confusion_matrix'])
    class_names = list(metrics['per_class_metrics'].keys())
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=str(output_path / 'confusion_matrix.png')
    )
    
    # Normalized confusion matrix
    print("2. Creating normalized confusion matrix...")
    plot_confusion_matrix_normalized(
        cm,
        class_names,
        save_path=str(output_path / 'confusion_matrix_normalized.png')
    )
    
    # Per-class accuracy
    print("3. Creating per-class accuracy plot...")
    plot_per_class_accuracy(
        metrics,
        save_path=str(output_path / 'per_class_accuracy.png')
    )
    
    # Metrics comparison
    print("4. Creating metrics comparison plot...")
    plot_metrics_comparison(
        metrics,
        save_path=str(output_path / 'metrics_comparison.png')
    )
    
    # Training history
    print("5. Creating training history plot...")
    if history_path is not None and Path(history_path).exists():
        plot_training_history(
            history_path,
            save_path=str(output_path / 'training_history.png')
        )
    else:
        print(f"   Skipping training history plot (path not provided or not found)")
    
    print(f"\n[OK] All visualizations saved to {output_dir}")


if __name__ == "__main__":
    """Example usage"""
    print("="*60)
    print("DeepSceneLoc - Visualization Utilities")
    print("Week 6 - Supporting Team")
    print("="*60)
    
    print("\nTo create all visualizations:")
    print("  from src.utils.visualizations import create_all_visualizations")
    print("  create_all_visualizations(")
    print("      metrics=eval_metrics,")
    print("      history_path='logs/training_history.json',")
    print("      output_dir='results/visualizations'")
    print("  )")