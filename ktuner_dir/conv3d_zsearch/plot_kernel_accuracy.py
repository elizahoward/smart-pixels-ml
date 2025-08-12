#!/usr/bin/env python3
"""
Script to plot training accuracy as a function of kernel size from Keras Tuner results.
Uses a pastel red color scheme.
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

def plot_kernel_accuracy_analysis():
    """
    Plot training accuracy as a function of kernel size.
    """
    # Set up pastel red color scheme
    pastel_reds = ['#ffcdd2', '#ef9a9a', '#e57373', '#ef5350', '#f44336', '#e53935']
    
    # Set style
    plt.style.use('default')
    sns.set_palette(pastel_reds)
    
    # Data collection
    kernel_data = []
    kernel_size_strs = []
    kernel_areas = []
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
            kernel_areas.append(kernel_area)
            accuracies.append(accuracy)
    
    if not kernel_data:
        print("No trial data found!")
        return
    
    # Convert to numpy arrays
    kernel_areas = np.array(kernel_areas)
    accuracies = np.array(accuracies)
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Accuracy vs Kernel Size Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Accuracy vs Kernel Area
    ax1.scatter(kernel_areas, accuracies, alpha=0.7, s=60, c=pastel_reds[2])
    ax1.set_xlabel('Kernel Area (rows × cols)', fontsize=12)
    ax1.set_ylabel('Validation Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Kernel Area', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(kernel_areas, accuracies, 1)
    p = np.poly1d(z)
    ax1.plot(kernel_areas, p(kernel_areas), "--", alpha=0.8, color=pastel_reds[4])
    
    # 2. Box plot grouped by kernel shape categories
    # Group kernels into categories
    square_kernels = []
    longer_width_kernels = []  # width > height (cols > rows)
    longer_length_kernels = []  # height > width (rows > cols)
    
    for i, kernel_size in enumerate(kernel_size_strs):
        rows, cols = map(int, kernel_size.split('×'))
        if rows == cols:
            square_kernels.append(accuracies[i])
        elif cols > rows:
            longer_width_kernels.append(accuracies[i])
        else:  # rows > cols
            longer_length_kernels.append(accuracies[i])
    
    # Create grouped data
    grouped_data = [square_kernels, longer_width_kernels, longer_length_kernels]
    group_labels = ['Square Kernels', 'Longer Width', 'Longer Length']
    
    # Create box plot
    box_plot = ax2.boxplot(grouped_data, labels=group_labels, patch_artist=True)
    
    # Color each group differently
    colors = [pastel_reds[1], pastel_reds[3], pastel_reds[5]]
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Kernel Shape Categories', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Accuracy Distribution by Kernel Shape', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Square: {len(square_kernels)} trials\nWidth: {len(longer_width_kernels)} trials\nLength: {len(longer_length_kernels)} trials"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Bar plot: Mean accuracy by kernel shape categories
    # Calculate means and stds for each group
    group_means = []
    group_stds = []
    
    for group_data in grouped_data:
        if group_data:  # Only if group has data
            group_means.append(np.mean(group_data))
            group_stds.append(np.std(group_data))
        else:
            group_means.append(0)
            group_stds.append(0)
    
    bars = ax3.bar(group_labels, group_means, yerr=group_stds, 
                   capsize=5, alpha=0.8, color=colors)
    
    # Add value labels on bars
    for bar, mean_acc in zip(bars, group_means):
        if mean_acc > 0:  # Only add label if there's data
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{mean_acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Kernel Shape Categories', fontsize=12)
    ax3.set_ylabel('Mean Validation Accuracy', fontsize=12)
    ax3.set_title('Mean Accuracy by Kernel Shape', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.8, 0.95)  # Adjust based on your data range
    
    # 4. Heatmap: Kernel rows vs cols
    kernel_rows_list = []
    kernel_cols_list = []
    
    for trial_data in kernel_data:
        kernel_size = trial_data['kernel_size']
        rows, cols = map(int, kernel_size.split('×'))
        kernel_rows_list.append(rows)
        kernel_cols_list.append(cols)
    
    # Create heatmap data
    unique_rows = sorted(list(set(kernel_rows_list)))
    unique_cols = sorted(list(set(kernel_cols_list)))
    
    heatmap_data = np.zeros((len(unique_rows), len(unique_cols)))
    count_data = np.zeros((len(unique_rows), len(unique_cols)))
    
    for i, (rows, cols, acc) in enumerate(zip(kernel_rows_list, kernel_cols_list, accuracies)):
        row_idx = unique_rows.index(rows)
        col_idx = unique_cols.index(cols)
        heatmap_data[row_idx, col_idx] += acc
        count_data[row_idx, col_idx] += 1
    
    # Average the accuracies
    for i in range(len(unique_rows)):
        for j in range(len(unique_cols)):
            if count_data[i, j] > 0:
                heatmap_data[i, j] /= count_data[i, j]
    
    im = ax4.imshow(heatmap_data, cmap='Reds', alpha=0.8)
    ax4.set_xticks(range(len(unique_cols)))
    ax4.set_yticks(range(len(unique_rows)))
    ax4.set_xticklabels(unique_cols)
    ax4.set_yticklabels(unique_rows)
    ax4.set_xlabel('Kernel Columns', fontsize=12)
    ax4.set_ylabel('Kernel Rows', fontsize=12)
    ax4.set_title('Mean Accuracy by Kernel Dimensions', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(unique_rows)):
        for j in range(len(unique_cols)):
            if count_data[i, j] > 0:
                text = ax4.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="white", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Mean Validation Accuracy', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = "kernel_accuracy_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("KERNEL SHAPE ANALYSIS SUMMARY")
    print("="*60)
    
    # Print statistics for each group
    for i, (group_name, group_data) in enumerate(zip(group_labels, grouped_data)):
        if group_data:
            mean_acc = np.mean(group_data)
            std_acc = np.std(group_data)
            count = len(group_data)
            print(f"{group_name}: {count} trials, Mean: {mean_acc:.4f} ± {std_acc:.4f}")
        else:
            print(f"{group_name}: 0 trials")
    
    # Find best performing group
    best_group_idx = np.argmax([np.mean(g) if g else 0 for g in grouped_data])
    best_group_name = group_labels[best_group_idx]
    best_group_mean = np.mean(grouped_data[best_group_idx]) if grouped_data[best_group_idx] else 0
    print(f"\nBest performing kernel shape: {best_group_name} (Mean accuracy: {best_group_mean:.4f})")
    
    # Print detailed breakdown of kernels in each category
    print("\n" + "-"*40)
    print("DETAILED KERNEL BREAKDOWN")
    print("-"*40)
    
    # Recreate kernel_accuracies for detailed breakdown
    kernel_accuracies = {}
    for i, kernel_size in enumerate(kernel_size_strs):
        if kernel_size not in kernel_accuracies:
            kernel_accuracies[kernel_size] = []
        kernel_accuracies[kernel_size].append(accuracies[i])
    
    # Print by category
    square_kernel_sizes = [k for k in kernel_accuracies.keys() if int(k.split('×')[0]) == int(k.split('×')[1])]
    width_kernel_sizes = [k for k in kernel_accuracies.keys() if int(k.split('×')[1]) > int(k.split('×')[0])]
    length_kernel_sizes = [k for k in kernel_accuracies.keys() if int(k.split('×')[0]) > int(k.split('×')[1])]
    
    print("Square kernels:", sorted(square_kernel_sizes))
    print("Longer width kernels:", sorted(width_kernel_sizes))
    print("Longer length kernels:", sorted(length_kernel_sizes))
    
    plt.show()

if __name__ == "__main__":
    plot_kernel_accuracy_analysis() 