#!/usr/bin/env python3
"""
Standalone script to generate plots from Model1 training results with trial averaging.
Creates validation comparison plots and best metrics bar charts with trial averaging.

Usage:
    python plot_model1_results.py <results_folder_path>
    
Examples:
    python plot_model1_results.py quantized_model1_results_20250130_120000/
    python plot_model1_results.py .  # To process all result directories
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from collections import defaultdict
import pandas as pd

class Model1ResultsPlotter:
    """
    Class to create plots from Model1 training results with trial averaging and blue color scheme.
    """
    
    def __init__(self, results_dir):
        """
        Initialize with results directory path.
        
        Args:
            results_dir (str or Path): Path to the results directory
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
            
        print(f"Loading results from: {self.results_dir}")
        
        # Use Set2 color scheme for line plots (matching Model2)
        self.set2_colors = plt.cm.Set2(np.linspace(0, 1, 10))
        # Use Reds color scheme for bar plots (matching Model2)
        self.reds_colors = plt.cm.Reds(np.linspace(0.3, 0.9, 10))  # Avoid very light and very dark
        # Custom red color (139, 0, 33) for histogram bars
        self.custom_red = (139/255, 0/255, 33/255)
        
    def detect_results_format(self):
        """
        Detect which type of results directory this is and return the appropriate format.
        
        Returns:
            str: 'model1' for Model1 results format
        """
        # Check for Model1 results format (model subdirectories with trials)
        model_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() 
                     and any(substring in d.name for substring in ['quantized_', 'non_quantized'])]
        
        if model_dirs:
            # Check if any model directory has trial subdirectories
            for model_dir in model_dirs:
                trial_dirs = [d for d in model_dir.iterdir() if d.is_dir() and 'trial' in d.name]
                if trial_dirs:
                    return 'model1'
                    
        return 'unknown'
    
    def load_model1_results_data(self):
        """
        Load data from Model1 results format (model subdirectories with trial subdirectories).
        
        Returns:
            dict: Dictionary containing averaged model data
        """
        print("Loading Model1 results data...")
        
        model_data = {}
        
        # Look for model subdirectories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            if not any(substring in model_dir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            print(f"  Processing: {model_dir.name}")
            
            # Load overall results if available
            overall_results_file = model_dir / "overall_results.json"
            if overall_results_file.exists():
                try:
                    with open(overall_results_file, 'r') as f:
                        overall_results = json.load(f)
                except Exception as e:
                    print(f"    Error loading overall results: {e}")
                    overall_results = {}
            else:
                overall_results = {}
            
            # Load averaged training history
            history_files = list(model_dir.glob("*_history.npz"))
            if not history_files:
                print(f"    Warning: No history file found in {model_dir.name}")
                continue
            
            history_file = history_files[0]
            
            try:
                # Load training history from .npz file
                history_data = np.load(history_file)
                history = {
                    'accuracy': history_data['accuracy'].tolist(),
                    'val_accuracy': history_data['val_accuracy'].tolist(),
                    'loss': history_data['loss'].tolist(),
                    'val_loss': history_data['val_loss'].tolist()
                }
                
                model_data[model_dir.name] = {
                    'history': history,
                    'eval_results': {
                        'roc_auc': overall_results.get('avg_roc_auc'),
                        'test_accuracy': overall_results.get('avg_test_accuracy'),
                        'test_loss': overall_results.get('avg_test_loss')
                    },
                    'dir_name': model_dir.name,
                    'n_trials': overall_results.get('n_trials', 1),
                    'weight_bits': overall_results.get('weight_bits'),
                    'model_type': overall_results.get('model_type', 'unknown')
                }
                
                print(f"    ✓ Loaded {len(history['val_accuracy'])} epochs of averaged data from {overall_results.get('n_trials', 1)} trials")
                
            except Exception as e:
                print(f"    Error loading data from {model_dir.name}: {e}")
                continue
        
        return model_data
    
    def create_validation_comparison_plots(self, model_data, save_dir):
        """
        Create validation accuracy and loss comparison plots matching Model2 style.
        
        Args:
            model_data (dict): Dictionary containing model data
            save_dir (Path): Directory to save plots
        """
        print("Creating validation comparison plots...")
        
        # Separate non-quantized and quantized models
        non_quantized_data = None
        quantized_data = []
        
        for name, data in model_data.items():
            if data['model_type'] == 'non_quantized':
                non_quantized_data = (name, data)
            else:
                quantized_data.append((name, data))
        
        # Sort quantized models by weight bits
        def extract_bit_width(name, data):
            return data.get('weight_bits', 999)
        
        quantized_data.sort(key=lambda x: extract_bit_width(x[0], x[1]))
        
        # Plot validation accuracy comparison
        plt.figure(figsize=(8.5, 8.5))  # Square plot matching Model2
        
        color_idx = 0
        
        # Plot quantized models first in Set2 color scheme (background)
        for name, data in quantized_data:
            val_acc = data['history']['val_accuracy']
            if val_acc:
                # Create display name
                bits = data.get('weight_bits', 'Unknown')
                display_name = f'{bits}-bit'
                n_trials = data.get('n_trials', 1)
                if n_trials > 1:
                    display_name += f" (avg of {n_trials} trials)"
                    
                color = self.set2_colors[color_idx % len(self.set2_colors)]
                plt.plot(val_acc, label=display_name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
        
        # Plot non-quantized model last in black (on top)
        if non_quantized_data:
            name, data = non_quantized_data
            val_acc = data['history']['val_accuracy']
            if val_acc:
                n_trials = data.get('n_trials', 1)
                label = 'Non-quantized'
                if n_trials > 1:
                    label += f" (avg of {n_trials} trials)"
                plt.plot(val_acc, label=label, color='black', linewidth=3, alpha=0.8, zorder=10)
        
        plt.title('Validation Accuracy Comparison (Trial Averaged)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Validation Accuracy', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        # Fix y-axis scaling - set reasonable limits to avoid squishing
        all_accuracies = []
        for name, data in model_data.items():
            val_acc = data['history']['val_accuracy']
            if val_acc:
                all_accuracies.extend(val_acc)
        
        if all_accuracies:
            min_acc = min(all_accuracies)
            max_acc = max(all_accuracies)
            margin = (max_acc - min_acc) * 0.15  # 15% margin for better spacing
            plt.ylim(max(0, min_acc - margin), min(1, max_acc + margin))
        
        plt.legend(fontsize=14, loc='best')
        plt.grid(True, alpha=0.3)
        
        acc_plot_file = save_dir / "validation_accuracy_comparison.png"
        plt.savefig(acc_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot validation loss comparison
        plt.figure(figsize=(8.5, 8.5))  # Square plot matching Model2
        
        color_idx = 0
        
        # Plot quantized models first in Set2 color scheme (background)
        for name, data in quantized_data:
            val_loss = data['history']['val_loss']
            if val_loss:
                bits = data.get('weight_bits', 'Unknown')
                display_name = f'{bits}-bit'
                n_trials = data.get('n_trials', 1)
                if n_trials > 1:
                    display_name += f" (avg of {n_trials} trials)"
                    
                color = self.set2_colors[color_idx % len(self.set2_colors)]
                plt.plot(val_loss, label=display_name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
        
        # Plot non-quantized model last in black (on top)
        if non_quantized_data:
            name, data = non_quantized_data
            val_loss = data['history']['val_loss']
            if val_loss:
                n_trials = data.get('n_trials', 1)
                label = 'Non-quantized'
                if n_trials > 1:
                    label += f" (avg of {n_trials} trials)"
                plt.plot(val_loss, label=label, color='black', linewidth=3, alpha=0.8, zorder=10)
        
        plt.title('Model1: Validation Loss Comparison (Trial Averaged)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        loss_plot_file = save_dir / "validation_loss_comparison.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: validation_accuracy_comparison.png")
        print(f"  ✓ Saved: validation_loss_comparison.png")
        
        return acc_plot_file, loss_plot_file
    
    def create_best_metrics_plots(self, model_data, save_dir):
        """
        Create a square bar plot for best validation accuracy with custom styling.
        
        Args:
            model_data (dict): Dictionary containing model data
            save_dir (Path): Directory to save plots
        """
        print("Creating best validation accuracy plot...")
        
        # Sort models: non-quantized first, then by weight bits
        sorted_models = []
        model_names = []
        
        for name, data in model_data.items():
            if data['model_type'] == 'non_quantized':
                sorted_models.append((name, data, 0))
                model_names.append('Float32')
            else:
                bits = data.get('weight_bits', 999)
                sorted_models.append((name, data, bits))
                model_names.append(f'{bits}-bit')
        
        # Sort by weight bits
        sorted_indices = sorted(range(len(sorted_models)), key=lambda i: sorted_models[i][2])
        sorted_models = [sorted_models[i] for i in sorted_indices]
        model_names = [model_names[i] for i in sorted_indices]
        
        # Extract best validation accuracy
        best_val_acc = []
        
        for name, data, _ in sorted_models:
            best_val_acc.append(max(data['history']['val_accuracy']))
        
        # Create taller figure
        fig, ax = plt.subplots(figsize=(9, 11))  # Made taller as requested
        
        x_pos = np.arange(len(model_names))
        
        # Create colors list - grey for Float32, custom red for others
        colors = []
        for name in model_names:
            if name == 'Float32':
                colors.append('grey')
            else:
                colors.append(self.custom_red)
        
        # Create bars with appropriate colors and thinner width
        bars = ax.bar(x_pos, best_val_acc, color=colors, edgecolor='black', linewidth=1, width=0.6)
        
        # Set labels and title with appropriate font sizes
        ax.set_ylabel('Best Validation Accuracy', fontsize=36)  # Increased from 24
        ax.set_title('Model1: Best Validation Accuracy', fontsize=28, fontweight='bold')  # Made smaller
        
        # Set tick marks and labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=30)  # Increased from 20
        ax.tick_params(axis='x', labelsize=30)  # X-axis tick mark size
        ax.tick_params(axis='y', labelsize=20)  # Smaller Y-axis tick mark size
        
        # Set y-axis limits for consistency
        ax.set_ylim(0.65, 1.1)
        
        # Add horizontal gridlines
        ax.grid(True, alpha=0.3, axis='y')
        
        # Hide the 1.1 tick label while keeping the tick
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        yticklabels = [f'{tick:.2f}' if abs(tick - 1.1) > 0.01 else '' for tick in yticks]
        ax.set_yticklabels(yticklabels)
        
        # Add value labels on top of bars with color matching bars (non-bold)
        for bar, val, color in zip(bars, best_val_acc, colors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=24, color=color)
        
        # Adjust layout to fit the larger fonts and give more space for title
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20, top=0.80, left=0.15, right=0.95)  # Even more space at top for title separation
        
        # Save the plot
        plt.savefig(save_dir / 'best_validation_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: best_validation_accuracy.png")
    
    def create_roc_curves_plot(self, save_dir):
        """
        Create ROC curves overlay plot if ROC data is available.
        
        Args:
            save_dir (Path): Directory to save plots
        """
        roc_file = self.results_dir / "all_roc_data.json"
        if not roc_file.exists():
            print("  No ROC data file found, skipping ROC curves plot")
            return
        
        print("Creating ROC curves overlay plot...")
        
        try:
            with open(roc_file, 'r') as f:
                roc_data = json.load(f)
        except Exception as e:
            print(f"  Error loading ROC data: {e}")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Sort by model type and bits
        sorted_roc_items = []
        for key, data in roc_data.items():
            if 'non_quantized' in key:
                sorted_roc_items.append((key, data, 0))
            else:
                # Extract weight bits from key (e.g., "quantized_8w0i_8a0i" -> 8)
                try:
                    bits = int(key.split('_')[1].split('w')[0])
                    sorted_roc_items.append((key, data, bits))
                except:
                    sorted_roc_items.append((key, data, 999))
        
        sorted_roc_items.sort(key=lambda x: x[2])
        
        for i, (key, data, _) in enumerate(sorted_roc_items):
            color = self.set2_colors[i % len(self.set2_colors)]
            plt.plot(data['fpr'], data['tpr'], 
                    color=color, linewidth=2, 
                    label=f"{data['model_name']} (AUC = {data['auc']:.3f})")
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Model1: ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'roc_curves_overlay.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: roc_curves_overlay.png")
    
    def create_summary_table(self, model_data, save_dir):
        """
        Create a summary table of all model results.
        
        Args:
            model_data (dict): Dictionary containing model data
            save_dir (Path): Directory to save the table
        """
        print("Creating summary table...")
        
        # Prepare data for table
        table_data = []
        
        for name, data in model_data.items():
            row = {
                'Model': 'Non-quantized' if data['model_type'] == 'non_quantized' else f"{data.get('weight_bits', 'Unknown')}-bit",
                'Weight Bits': 'N/A' if data['model_type'] == 'non_quantized' else data.get('weight_bits', 'Unknown'),
                'Trials': data.get('n_trials', 1),
                'Best Val Accuracy': f"{max(data['history']['val_accuracy']):.4f}",
                'Best Val Loss': f"{min(data['history']['val_loss']):.4f}",
                'Test Accuracy': f"{data['eval_results']['test_accuracy']:.4f}" if data['eval_results']['test_accuracy'] else 'N/A',
                'ROC AUC': f"{data['eval_results']['roc_auc']:.4f}" if data['eval_results']['roc_auc'] else 'N/A'
            }
            table_data.append(row)
        
        # Sort by weight bits
        table_data.sort(key=lambda x: 0 if x['Weight Bits'] == 'N/A' else int(x['Weight Bits']) if x['Weight Bits'] != 'Unknown' else 999)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        df.to_csv(save_dir / 'model1_results_summary.csv', index=False)
        
        # Save as formatted text
        with open(save_dir / 'model1_results_summary.txt', 'w') as f:
            f.write("Model1 Training Results Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(df.to_string(index=False))
            f.write(f"\n\nGenerated on: {save_dir.parent.name}\n")
        
        print(f"  ✓ Saved: model1_results_summary.csv")
        print(f"  ✓ Saved: model1_results_summary.txt")
        
        # Print summary to console
        print("\nModel1 Results Summary:")
        print("-" * 80)
        print(df.to_string(index=False))
    
    def create_val_acc_and_roc_auc_chart(self, model_data, save_dir):
        """
        Create a double y-axis bar chart showing validation accuracy and ROC AUC with trial averaging.
        Validation accuracy is shown in red (front), ROC AUC in blue (back).
        
        Args:
            model_data (dict): Dictionary containing model data
            save_dir (Path): Directory to save plots
        """
        print("Creating validation accuracy and ROC AUC double y-axis chart (trial averaged)...")
        
        # Prepare data
        model_names = []
        best_val_accuracies = []
        roc_aucs = []
        n_trials_list = []
        
        # Separate non-quantized and quantized models
        non_quantized_data = None
        quantized_data = []
        
        for name, data in model_data.items():
            if data['model_type'] == 'non_quantized':
                non_quantized_data = (name, data)
            else:
                quantized_data.append((name, data))
        
        # Sort quantized models by weight bits
        def extract_bit_width(name, data):
            return data.get('weight_bits', 999)
        
        quantized_data.sort(key=lambda x: extract_bit_width(x[0], x[1]))
        
        # Add non-quantized model first
        if non_quantized_data:
            name, data = non_quantized_data
            val_acc = data['history']['val_accuracy']
            eval_results = data.get('eval_results', {})
            roc_auc = eval_results.get('roc_auc', None)
            n_trials = data.get('n_trials', 1)
            
            if val_acc and roc_auc is not None:
                label = 'Non-quantized'
                model_names.append(label)
                best_val_accuracies.append(max(val_acc))
                roc_aucs.append(roc_auc)
                n_trials_list.append(n_trials)
        
        # Add quantized models
        for name, data in quantized_data:
            val_acc = data['history']['val_accuracy']
            eval_results = data.get('eval_results', {})
            roc_auc = eval_results.get('roc_auc', None)
            n_trials = data.get('n_trials', 1)
            
            if val_acc and roc_auc is not None:
                bits = data.get('weight_bits', 'Unknown')
                display_name = f'{bits}-bit'
                model_names.append(display_name)
                best_val_accuracies.append(max(val_acc))
                roc_aucs.append(roc_auc)
                n_trials_list.append(n_trials)
        
        if not model_names:
            print("Warning: No models with both validation accuracy and ROC AUC data found")
            return None
        
        # Create figure with double y-axis
        fig, ax1 = plt.subplots(figsize=(7.0, 7.0))  # Square plot
        
        x_pos = np.arange(len(model_names))
        
        # Create second y-axis for ROC AUC
        ax2 = ax1.twinx()
        
        # Plot AUC bars in light translucent grey FIRST (background) - wider bars behind
        bars2 = ax2.bar(x_pos, roc_aucs, alpha=0.4, color='#808080', 
                       label='ROC AUC', width=0.8, zorder=1)
        
        # Calculate AUC axis limits with standard auto-scaling FIRST
        if roc_aucs:
            min_auc = min(roc_aucs)
            max_auc = max(roc_aucs)
            margin = (max_auc - min_auc) * 0.1  # 10% margin
            auc_lower = min_auc - margin
            auc_upper = max_auc + margin
            ax2.set_ylim(auc_lower, auc_upper)
        
        # Add value labels on AUC bars (positioned 5% from top of graph border)
        if roc_aucs:
            # Get the FINAL AUC axis limits and position labels 5% from the top
            auc_lower, auc_upper = ax2.get_ylim()
            label_height = auc_upper - (auc_upper - auc_lower) * 0.05  # 5% from top
            
            for i, (bar, val) in enumerate(zip(bars2, roc_aucs)):
                # Position AUC labels at a consistent height 5% from top of graph
                ax2.text(bar.get_x() + bar.get_width() * 0.5, label_height,
                        f'{val:.3f}', ha='center', va='center', fontsize=8, 
                        color='#404040', fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Plot validation accuracy bars in red SECOND (foreground) - narrower bars in front
        bars1 = ax1.bar(x_pos, best_val_accuracies, alpha=0.9, color='#d62728', 
                       edgecolor='black', linewidth=1.0, label='Best Val Accuracy', 
                       width=0.5, zorder=3)
        
        # Add value labels on validation accuracy bars
        for i, (bar, val) in enumerate(zip(bars1, best_val_accuracies)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, 
                    color='#d62728', fontweight='bold')
        
        # Set labels and title
        ax1.set_ylabel('Best Validation Accuracy', color='#d62728', fontsize=12)
        ax2.set_ylabel('ROC AUC', color='#808080', fontsize=12)
        ax1.set_title('Model1: Validation Accuracy and ROC AUC (Averaged)', fontsize=16, fontweight='bold')
        
        # Show model names on x-axis with better formatting
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
        
        # Set y-axis limits for validation accuracy to focus on the range with extra space for labels
        if best_val_accuracies:
            min_acc = min(best_val_accuracies)
            max_acc = max(best_val_accuracies)
            margin = (max_acc - min_acc) * 0.15  # 15% margin for label space
            ax1.set_ylim(max(0, min_acc - margin), min(1, max_acc + margin + 0.02))  # Extra space at top
        
        # Color the y-axis labels to match the data
        ax1.tick_params(axis='y', labelcolor='#d62728')
        ax2.tick_params(axis='y', labelcolor='#808080')
        
        # Add grids
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Create combined legend on ax2 (second axis renders on top)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10, 
                  framealpha=1.0, facecolor='white', edgecolor='black')
        
        plt.tight_layout()
        # Adjust layout to reduce bottom whitespace
        plt.subplots_adjust(bottom=0.15)
        
        # Save the plot
        combined_plot_file = save_dir / "best_val_acc_and_roc_auc.png"
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: best_val_acc_and_roc_auc.png")
        
        return combined_plot_file
    
    def generate_all_plots(self):
        """
        Generate all plots for the results directory.
        """
        print(f"\n=== Generating Model1 Results Plots ===")
        
        # Detect format
        format_type = self.detect_results_format()
        if format_type == 'unknown':
            print("❌ Unknown results format. Expected Model1 format with model subdirectories.")
            return
        
        print(f"Detected format: {format_type}")
        
        # Load data
        if format_type == 'model1':
            model_data = self.load_model1_results_data()
        else:
            print(f"❌ Unsupported format: {format_type}")
            return
        
        if not model_data:
            print("❌ No model data found!")
            return
        
        print(f"\nFound {len(model_data)} model configurations:")
        for name in model_data.keys():
            print(f"  - {name}")
        
        # Create plots
        save_dir = self.results_dir
        
        acc_plot, loss_plot = self.create_validation_comparison_plots(model_data, save_dir)
        self.create_best_metrics_plots(model_data, save_dir)
        self.create_roc_curves_plot(save_dir)
        combined_chart = self.create_val_acc_and_roc_auc_chart(model_data, save_dir)
        self.create_summary_table(model_data, save_dir)
        
        print(f"\n✅ All plots generated successfully!")
        print(f"Results saved in: {save_dir}")
        
        # List generated files
        plot_files = ['validation_accuracy_comparison.png', 'validation_loss_comparison.png',
                     'best_validation_accuracy.png', 'best_val_acc_and_roc_auc.png',
                     'roc_curves_overlay.png', 'model1_results_summary.csv', 
                     'model1_results_summary.txt']
        
        print("\nGenerated files:")
        for file in plot_files:
            if (save_dir / file).exists():
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (not generated)")

def main():
    """Main function to run the plotting script"""
    parser = argparse.ArgumentParser(description='Generate plots from Model1 training results')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--output', help='Output directory (default: same as results_dir)')
    
    args = parser.parse_args()
    
    try:
        plotter = Model1ResultsPlotter(args.results_dir)
        plotter.generate_all_plots()
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()