#!/usr/bin/env python3
"""
Script to plot training accuracy grouped by kernel shape categories.
Generates a single plot with legend at bottom right and max values marked.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_trial_data(trial_dir):
    """
    Load trial data from a trial directory.
    
    Args:
        trial_dir (Path): Path to trial directory
        
    Returns:
        dict: Trial data with hyperparameters and metrics
    """
    trial_file = trial_dir / "trial.json"
    if not trial_file.exists():
        return None
    
    with open(trial_file, 'r') as f:
        trial_data = json.load(f)
    
    return trial_data

def extract_kernel_size_and_accuracy(trial_data):
    """
    Extract kernel size and accuracy from trial data.
    
    Args:
        trial_data (dict): Trial data from JSON
        
    Returns:
        tuple: (kernel_size_str, kernel_area, accuracy)
    """
    if not trial_data:
        return None
    
    # Extract hyperparameters
    hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
    kernel_rows = hyperparams.get('kernel_rows', 3)
    kernel_cols = hyperparams.get('kernel_cols', 3)
    
    # Extract accuracy (use validation accuracy as it's more reliable)
    metrics = trial_data.get('metrics', {}).get('metrics', {})
    val_accuracy = metrics.get('val_accuracy', {}).get('observations', [])
    
    if not val_accuracy:
        return None
    
    # Get the best accuracy (last observation)
    accuracy = val_accuracy[-1]['value'][0]
    
    # Create kernel size string and calculate area
    kernel_size_str = f"{kernel_rows}×{kernel_cols}"
    kernel_area = kernel_rows * kernel_cols
    
    return kernel_size_str, kernel_area, accuracy

def plot_kernel_shape_groups():
    """
    Plot training accuracy grouped by kernel shape categories.
    """
    # Set up pastel red color scheme
    pastel_reds = ['#ffcdd2', '#ef9a9a', '#e57373', '#ef5350', '#f44336', '#e53935']
    
    # Set style
    plt.style.use('default')
    sns.set_palette(pastel_reds)
    
    # Data collection
    kernel_data = []
    kernel_size_strs = []
    accuracies = []
    
    # Load data from all trials
    base_dir = Path(".")
    
    for trial_dir in sorted(base_dir.glob("trial_*")):
        trial_data = load_trial_data(trial_dir)
        result = extract_kernel_size_and_accuracy(trial_data)
        
        if result:
            kernel_size_str, kernel_area, accuracy = result
            kernel_data.append({
                'kernel_size': kernel_size_str,
                'kernel_area': kernel_area,
                'accuracy': accuracy
            })
            kernel_size_strs.append(kernel_size_str)
            accuracies.append(accuracy)
    
    if not kernel_data:
        print("No trial data found!")
        return
    
    # Convert to numpy arrays
    accuracies = np.array(accuracies)
    
    # Group kernels into categories
    square_kernels = []
    longer_width_kernels = []  # width > height (cols > rows)
    longer_length_kernels = []  # height > width (rows > cols)
    
    # Also track which specific kernels belong to each group for max identification
    square_kernel_sizes = []
    width_kernel_sizes = []
    length_kernel_sizes = []
    
    for i, kernel_size in enumerate(kernel_size_strs):
        rows, cols = map(int, kernel_size.split('×'))
        if rows == cols:
            square_kernels.append(accuracies[i])
            square_kernel_sizes.append(kernel_size)
        elif cols > rows:
            longer_width_kernels.append(accuracies[i])
            width_kernel_sizes.append(kernel_size)
        else:  # rows > cols
            longer_length_kernels.append(accuracies[i])
            length_kernel_sizes.append(kernel_size)
    
    # Create grouped data
    grouped_data = [square_kernels, longer_width_kernels, longer_length_kernels]
    group_labels = ['Square Kernels', 'Longer Width', 'Longer Length']
    group_kernel_sizes = [square_kernel_sizes, width_kernel_sizes, length_kernel_sizes]
    
    # Create figure (20% smaller)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.4))
    
    # Create box plot
    colors = [pastel_reds[1], pastel_reds[3], pastel_reds[5]]
    box_plot = ax.boxplot(grouped_data, labels=group_labels, patch_artist=True)
    
    # Color each group differently
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Find max values for each group and mark them
    max_values = []
    max_kernels = []
    
    for i, (group_data, kernel_sizes) in enumerate(zip(grouped_data, group_kernel_sizes)):
        if group_data:
            max_idx = np.argmax(group_data)
            max_val = group_data[max_idx]
            max_kernel = kernel_sizes[max_idx]
            max_values.append(max_val)
            max_kernels.append(max_kernel)
            
            # Mark the maximum point
            ax.plot(i+1, max_val, 'o', color='darkred', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            ax.annotate(f'{max_kernel}\n{max_val:.4f}', 
                       xy=(i+1, max_val), xytext=(10, 10),
                       textcoords='offset points', ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=12, fontweight='bold')
        else:
            max_values.append(0)
            max_kernels.append('N/A')
    
    # Add statistics to legend
    legend_text = []
    for i, (group_name, group_data, max_val, max_kernel) in enumerate(zip(group_labels, grouped_data, max_values, max_kernels)):
        if group_data:
            mean_acc = np.mean(group_data)
            std_acc = np.std(group_data)
            count = len(group_data)
            legend_text.append(f'{group_name}\n{count} trials, Mean: {mean_acc:.4f} ± {std_acc:.4f}\nMax: {max_kernel} ({max_val:.4f})')
        else:
            legend_text.append(f'{group_name}\n0 trials')
    
    # Create custom legend
    legend_elements = []
    for i, (color, text) in enumerate(zip(colors, legend_text)):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=text))
    
    # Place legend at bottom left
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    
    # Customize plot with larger labels
    ax.set_xlabel('Kernel Shape Categories', fontsize=16, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=16, fontweight='bold')
    ax.set_title('Sample Kernel Optimisation', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 0.95)  # Adjust based on your data range
    
    # Add some padding for the legend
    plt.tight_layout()
    
    # Save the plot
    output_file = "kernel_shape_groups_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved as: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("KERNEL SHAPE GROUPS ANALYSIS")
    print("="*60)
    
    for i, (group_name, group_data, max_val, max_kernel) in enumerate(zip(group_labels, grouped_data, max_values, max_kernels)):
        if group_data:
            mean_acc = np.mean(group_data)
            std_acc = np.std(group_data)
            count = len(group_data)
            print(f"{group_name}: {count} trials, Mean: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"  Max: {max_kernel} ({max_val:.4f})")
        else:
            print(f"{group_name}: 0 trials")
    
    # Find best performing group
    best_group_idx = np.argmax([np.mean(g) if g else 0 for g in grouped_data])
    best_group_name = group_labels[best_group_idx]
    best_group_mean = np.mean(grouped_data[best_group_idx]) if grouped_data[best_group_idx] else 0
    print(f"\nBest performing kernel shape: {best_group_name} (Mean accuracy: {best_group_mean:.4f})")
    
    plt.show()

if __name__ == "__main__":
    plot_kernel_shape_groups() 