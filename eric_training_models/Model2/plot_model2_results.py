#!/usr/bin/env python3
"""
Standalone script to generate plots from Model2 training results with trial averaging.
Creates validation comparison plots and best metrics bar charts with trial averaging.

Usage:
    python plot_model2_results.py <results_folder_path>
    
Examples:
    python plot_model2_results.py quantized_complicated_results_20250730_030326/
    python plot_model2_results.py results_2000unshuffled/
    python plot_model2_results.py .  # To process all result directories
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

class Model2ResultsPlotter:
    """
    Class to create plots from Model2 training results with trial averaging and red color scheme.
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
        
        # Use Set2 color scheme for line plots
        self.set2_colors = plt.cm.Set2(np.linspace(0, 1, 10))
        # Use Reds color scheme for bar plots
        self.reds_colors = plt.cm.Reds(np.linspace(0.3, 0.9, 10))  # Avoid very light and very dark
        # Custom red color (139, 0, 33) for histogram bars
        self.custom_red = (139/255, 0/255, 33/255)
        
    def detect_results_format(self):
        """
        Detect which type of results directory this is and return the appropriate format.
        
        Returns:
            str: 'complicated', 'unshuffled', or 'simple'
        """
        # Check for complicated results format (model subdirectories with trials)
        model_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() 
                     and any(substring in d.name for substring in ['quantized_', 'non_quantized'])]
        
        if model_dirs:
            # Check if any model directory has trial subdirectories
            for model_dir in model_dirs:
                trial_dirs = [d for d in model_dir.iterdir() if d.is_dir() and 'trial' in d.name]
                if trial_dirs:
                    return 'complicated'
                    
        # Check for unshuffled results format (individual trial files)
        trial_files = list(self.results_dir.glob("*trial*.h5")) or list(self.results_dir.glob("*trial*.npz"))
        if trial_files:
            return 'unshuffled'
            
        # Check for simple format (single history file)
        if (self.results_dir / "training_history.npz").exists():
            return 'simple'
            
        return 'unknown'
    
    def load_complicated_results_data(self):
        """
        Load data from complicated results format (model subdirectories with trial subdirectories).
        
        Returns:
            dict: Dictionary containing averaged model data
        """
        print("Loading complicated results data...")
        
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
            history_file = model_dir / f"{model_dir.name}_history.npz"
            if not history_file.exists():
                # Try alternative naming
                history_files = list(model_dir.glob("*_history.npz"))
                if history_files:
                    history_file = history_files[0]
                else:
                    print(f"    Warning: No history file found in {model_dir.name}")
                    continue
            
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
                    'n_trials': overall_results.get('n_trials', 1)
                }
                
                print(f"    ✓ Loaded {len(history['val_accuracy'])} epochs of averaged data from {overall_results.get('n_trials', 1)} trials")
                
            except Exception as e:
                print(f"    Error loading data from {model_dir.name}: {e}")
                continue
        
        return model_data
    
    def load_unshuffled_results_data(self):
        """
        Load data from unshuffled results format (individual trial files in one directory).
        Average across trials for each configuration.
        
        Returns:
            dict: Dictionary containing averaged model data
        """
        print("Loading unshuffled results data...")
        
        # Load summary CSV if available
        csv_file = self.results_dir / "quantized_model_results.csv"
        summary_data = {}
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    key = f"quantized_{int(row['weight_bits'])}bit_int{int(row['integer_bits'])}"
                    summary_data[key] = {
                        'avg_test_accuracy': row.get('final_val_accuracy'),
                        'avg_test_loss': row.get('final_val_loss'),
                        'avg_roc_auc': row.get('auc_score'),
                        'n_trials': int(row.get('n_trials', 2))
                    }
            except Exception as e:
                print(f"    Warning: Could not load CSV summary: {e}")
        
        # Group trial files by configuration
        trial_groups = defaultdict(list)
        
        for file_path in self.results_dir.glob("*_history.npz"):
            # Extract configuration from filename
            filename = file_path.stem
            # Remove trial suffix if present
            if filename.endswith('_trial1') or filename.endswith('_trial2'):
                config_name = filename.rsplit('_trial', 1)[0]
            else:
                config_name = filename.replace('_history', '')
            
            trial_groups[config_name].append(file_path)
        
        model_data = {}
        
        for config_name, trial_files in trial_groups.items():
            print(f"  Processing: {config_name} ({len(trial_files)} trials)")
            
            # Load all trials for this configuration
            all_histories = []
            for trial_file in trial_files:
                try:
                    trial_data = np.load(trial_file)
                    trial_history = {
                        'accuracy': trial_data['accuracy'],
                        'val_accuracy': trial_data['val_accuracy'],
                        'loss': trial_data['loss'],
                        'val_loss': trial_data['val_loss']
                    }
                    all_histories.append(trial_history)
                except Exception as e:
                    print(f"    Warning: Could not load {trial_file}: {e}")
                    continue
            
            if not all_histories:
                continue
            
            # Average across trials
            try:
                # Find the minimum length across all trials
                min_length = min(len(h['val_accuracy']) for h in all_histories)
                
                # Truncate all histories to the minimum length and average
                averaged_history = {}
                for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                    values = np.array([h[key][:min_length] for h in all_histories])
                    averaged_history[key] = np.mean(values, axis=0).tolist()
                
                # Get evaluation results from summary if available
                eval_results = summary_data.get(config_name, {})
                
                model_data[config_name] = {
                    'history': averaged_history,
                    'eval_results': {
                        'roc_auc': eval_results.get('avg_roc_auc'),
                        'test_accuracy': eval_results.get('avg_test_accuracy'),
                        'test_loss': eval_results.get('avg_test_loss')
                    },
                    'dir_name': config_name,
                    'n_trials': len(all_histories)
                }
                
                print(f"    ✓ Loaded {min_length} epochs averaged from {len(all_histories)} trials")
                
            except Exception as e:
                print(f"    Error averaging data for {config_name}: {e}")
                continue
        
        return model_data
    
    def load_simple_results_data(self):
        """
        Load data from simple results format (single training_history.npz file).
        
        Returns:
            dict: Dictionary containing model data
        """
        print("Loading simple results data...")
        
        history_file = self.results_dir / "training_history.npz"
        
        try:
            history_data = np.load(history_file)
            history = {
                'accuracy': history_data['accuracy'].tolist(),
                'val_accuracy': history_data['val_accuracy'].tolist(),
                'loss': history_data['loss'].tolist(),
                'val_loss': history_data['val_loss'].tolist()
            }
            
            model_data = {
                'single_model': {
                    'history': history,
                    'eval_results': {},
                    'dir_name': 'single_model',
                    'n_trials': 1
                }
            }
            
            print(f"    ✓ Loaded {len(history['val_accuracy'])} epochs of data")
            return model_data
            
        except Exception as e:
            print(f"    Error loading simple data: {e}")
            return {}
    
    def load_model_data(self):
        """
        Load training history and evaluation data from the results directory.
        Automatically detects the format and loads appropriately.
        
        Returns:
            dict: Dictionary containing model data organized by model type
        """
        format_type = self.detect_results_format()
        print(f"Detected results format: {format_type}")
        
        if format_type == 'complicated':
            model_data = self.load_complicated_results_data()
        elif format_type == 'unshuffled':
            model_data = self.load_unshuffled_results_data()
        elif format_type == 'simple':
            model_data = self.load_simple_results_data()
        else:
            print("Warning: Unknown results format, trying all methods...")
            model_data = {}
            
            # Try complicated format first
            try:
                model_data.update(self.load_complicated_results_data())
            except:
                pass
                
            # Try unshuffled format
            try:
                model_data.update(self.load_unshuffled_results_data())
            except:
                pass
                
            # Try simple format
            try:
                model_data.update(self.load_simple_results_data())
            except:
                pass
        
        print(f"Successfully loaded data from {len(model_data)} model configurations")
        return model_data
    
    def create_validation_comparison_plots(self, model_data):
        """
        Create validation accuracy and loss comparison plots with Set2 color scheme.
        
        Args:
            model_data (dict): Dictionary containing model data
        """
        print("\nCreating validation comparison plots...")
        
        # Separate non-quantized and quantized models
        non_quantized_data = None
        quantized_data = []
        
        for model_name, data in model_data.items():
            if 'non_quantized' in model_name.lower():
                non_quantized_data = (model_name, data)
            else:
                quantized_data.append((model_name, data))
        
        # Sort quantized models by bit width for consistent ordering
        def extract_bit_width(model_name):
            try:
                # Extract bit width from model name (e.g., "quantized_2bit_int0" -> 2)
                parts = model_name.split('_')
                for part in parts:
                    if 'bit' in part:
                        return int(part.replace('bit', ''))
                return 0
            except:
                return 0
        
        quantized_data.sort(key=lambda x: extract_bit_width(x[0]))
        
        # Plot validation accuracy comparison
        plt.figure(figsize=(8.5, 8.5))  # Square plot
        
        color_idx = 0
        
        # Plot quantized models first in Set2 color scheme (background)
        for model_name, data in quantized_data:
            val_acc = data['history'].get('val_accuracy', [])
            if val_acc:
                # Create display name
                display_name = self.format_model_name(model_name)
                n_trials = data.get('n_trials', 1)
                if n_trials > 1:
                    display_name += f" (avg of {n_trials} trials)"
                    
                color = self.set2_colors[color_idx % len(self.set2_colors)]
                plt.plot(val_acc, label=display_name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
        
        # Plot non-quantized model last in black (on top)
        if non_quantized_data:
            name, data = non_quantized_data
            val_acc = data['history'].get('val_accuracy', [])
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
        for model_name, data in model_data.items():
            val_acc = data['history'].get('val_accuracy', [])
            if val_acc:
                all_accuracies.extend(val_acc)
        
        if all_accuracies:
            min_acc = min(all_accuracies)
            max_acc = max(all_accuracies)
            margin = (max_acc - min_acc) * 0.15  # 15% margin for better spacing
            plt.ylim(max(0, min_acc - margin), min(1, max_acc + margin))
        
        plt.legend(fontsize=14, loc='best')
        plt.grid(True, alpha=0.3)
        
        acc_plot_file = self.results_dir / "validation_accuracy_comparison.png"
        plt.savefig(acc_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot validation loss comparison
        plt.figure(figsize=(8.5, 8.5))  # Square plot
        
        color_idx = 0
        
        # Plot quantized models first in Set2 color scheme (background)
        for model_name, data in quantized_data:
            val_loss = data['history'].get('val_loss', [])
            if val_loss:
                display_name = self.format_model_name(model_name)
                n_trials = data.get('n_trials', 1)
                if n_trials > 1:
                    display_name += f" (avg of {n_trials} trials)"
                    
                color = self.set2_colors[color_idx % len(self.set2_colors)]
                plt.plot(val_loss, label=display_name, color=color, linewidth=2, alpha=0.8)
                color_idx += 1
        
        # Plot non-quantized model last in black (on top)
        if non_quantized_data:
            name, data = non_quantized_data
            val_loss = data['history'].get('val_loss', [])
            if val_loss:
                n_trials = data.get('n_trials', 1)
                label = 'Non-quantized'
                if n_trials > 1:
                    label += f" (avg of {n_trials} trials)"
                plt.plot(val_loss, label=label, color='black', linewidth=3, alpha=0.8, zorder=10)
        
        plt.title('Validation Loss Comparison (Trial Averaged)', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        loss_plot_file = self.results_dir / "validation_loss_comparison.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Validation accuracy comparison saved to: {acc_plot_file}")
        print(f"✓ Validation loss comparison saved to: {loss_plot_file}")
        
        return acc_plot_file, loss_plot_file
    
    def create_best_metrics_bar_charts(self, model_data):
        """
        Create bar charts for best validation accuracy and loss, plus ROC AUC.
        
        Args:
            model_data (dict): Dictionary containing model data
        """
        print("\nCreating best metrics bar charts...")
        
        # Prepare data
        model_names = []
        best_val_accuracies = []
        best_val_losses = []
        roc_aucs = []
        n_trials_list = []
        colors = []
        
        # Separate non-quantized and quantized models
        non_quantized_data = None
        quantized_data = []
        
        for model_name, data in model_data.items():
            if 'non_quantized' in model_name.lower():
                non_quantized_data = (model_name, data)
            else:
                quantized_data.append((model_name, data))
        
        # Sort quantized models by bit width
        def extract_bit_width(model_name):
            try:
                parts = model_name.split('_')
                for part in parts:
                    if 'bit' in part:
                        return int(part.replace('bit', ''))
                return 0
            except:
                return 0
        
        quantized_data.sort(key=lambda x: extract_bit_width(x[0]))
        
        # Add non-quantized model first
        if non_quantized_data:
            name, data = non_quantized_data
            val_acc = data['history'].get('val_accuracy', [])
            val_loss = data['history'].get('val_loss', [])
            eval_results = data.get('eval_results', {})
            roc_auc = eval_results.get('roc_auc', None)
            n_trials = data.get('n_trials', 1)
            
            if val_acc and val_loss:
                model_names.append('Non-quantized')
                best_val_accuracies.append(max(val_acc))
                best_val_losses.append(min(val_loss))
                roc_aucs.append(roc_auc if roc_auc is not None else 0)
                n_trials_list.append(n_trials)
                colors.append('#404040')  # Dark grey
        
        # Add quantized models
        color_idx = 0
        for model_name, data in quantized_data:
            val_acc = data['history'].get('val_accuracy', [])
            val_loss = data['history'].get('val_loss', [])
            eval_results = data.get('eval_results', {})
            roc_auc = eval_results.get('roc_auc', None)
            n_trials = data.get('n_trials', 1)
            
            if val_acc and val_loss:
                display_name = self.format_model_name(model_name)
                model_names.append(display_name)
                best_val_accuracies.append(max(val_acc))
                best_val_losses.append(min(val_loss))
                roc_aucs.append(roc_auc if roc_auc is not None else 0)
                n_trials_list.append(n_trials)
                colors.append(self.reds_colors[color_idx % len(self.reds_colors)])
                color_idx += 1
        
        # Create separate bar chart for validation accuracy
        plt.figure(figsize=(7.0, 7.0))  # Square plot
        bars = plt.bar(model_names, best_val_accuracies, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.0)
        
        plt.title('Best Validation Accuracy by Model (Trial Averaged)', fontsize=16, fontweight='bold')
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Best Validation Accuracy', fontsize=12)
        
        # Set y-axis limits to focus on the range of values for better comparison
        if best_val_accuracies:
            min_acc = min(best_val_accuracies)
            max_acc = max(best_val_accuracies)
            margin = (max_acc - min_acc) * 0.1  # 10% margin
            plt.ylim(max(0, min_acc - margin), min(1, max_acc + margin))
        
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        acc_bar_file = self.results_dir / "best_validation_accuracy.png"
        plt.savefig(acc_bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate bar chart for validation loss
        plt.figure(figsize=(7.0, 7.0))  # Square plot
        bars = plt.bar(model_names, best_val_losses, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.0)
        
        plt.title('Best Validation Loss by Model (Trial Averaged)', fontsize=16, fontweight='bold')
        plt.xlabel('Model Type', fontsize=12)
        plt.ylabel('Best Validation Loss', fontsize=12)
        
        # Set y-axis limits to focus on the range of values for better comparison
        if best_val_losses:
            min_loss = min(best_val_losses)
            max_loss = max(best_val_losses)
            margin = (max_loss - min_loss) * 0.1  # 10% margin
            plt.ylim(min_loss - margin, max_loss + margin)
        
        plt.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        loss_bar_file = self.results_dir / "best_validation_loss.png"
        plt.savefig(loss_bar_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create bar chart for ROC AUC if available
        if any(roc_aucs):
            plt.figure(figsize=(7.0, 7.0))  # Square plot
            bars = plt.bar(model_names, roc_aucs, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.0)
            
            plt.title('ROC AUC by Model (Trial Averaged)', fontsize=16, fontweight='bold')
            plt.xlabel('Model Type', fontsize=12)
            plt.ylabel('ROC AUC', fontsize=12)
            
            # Set y-axis limits to focus on the range of values for better comparison
            valid_aucs = [auc for auc in roc_aucs if auc > 0]
            if valid_aucs:
                min_auc = min(valid_aucs)
                max_auc = max(valid_aucs)
                margin = (max_auc - min_auc) * 0.1  # 10% margin
                plt.ylim(min_auc - margin, max_auc + margin)
            
            plt.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            auc_bar_file = self.results_dir / "roc_auc_comparison.png"
            plt.savefig(auc_bar_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ ROC AUC bar chart saved to: {auc_bar_file}")
        
        print(f"✓ Best validation accuracy bar chart saved to: {acc_bar_file}")
        print(f"✓ Best validation loss bar chart saved to: {loss_bar_file}")
        
        # Print summary
        print("\n" + "=" * 90)
        print("BEST METRICS SUMMARY (TRIAL AVERAGED)")
        print("=" * 90)
        print(f"{'Model Type':<25} {'Best Val Acc':<15} {'Best Val Loss':<15} {'ROC AUC':<15} {'# Trials':<10}")
        print("-" * 90)
        for name, acc, loss, auc, n_trials in zip(model_names, best_val_accuracies, best_val_losses, roc_aucs, n_trials_list):
            auc_str = f"{auc:.4f}" if auc > 0 else "N/A"
            print(f"{name:<25} {acc:.4f}          {loss:.4f}          {auc_str:<15} {n_trials}")
        
        if best_val_accuracies and best_val_losses:
            best_acc_idx = np.argmax(best_val_accuracies)
            best_loss_idx = np.argmin(best_val_losses)
            print(f"\nBest validation accuracy: {model_names[best_acc_idx]} ({best_val_accuracies[best_acc_idx]:.4f})")
            print(f"Best validation loss: {model_names[best_loss_idx]} ({best_val_losses[best_loss_idx]:.4f})")
        
        valid_aucs = [(i, auc) for i, auc in enumerate(roc_aucs) if auc > 0]
        if valid_aucs:
            best_auc_idx, best_auc = max(valid_aucs, key=lambda x: x[1])
            print(f"Best ROC AUC: {model_names[best_auc_idx]} ({best_auc:.4f})")
        
        return acc_bar_file, loss_bar_file
    
    def create_val_acc_and_roc_auc_chart(self, model_data):
        """
        Create a double y-axis bar chart showing validation accuracy and ROC AUC with trial averaging.
        Validation accuracy is shown in red (front), ROC AUC in blue (back).
        
        Args:
            model_data (dict): Dictionary containing model data
        """
        print("\nCreating validation accuracy and ROC AUC double y-axis chart (trial averaged)...")
        
        # Prepare data
        model_names = []
        best_val_accuracies = []
        roc_aucs = []
        n_trials_list = []
        
        # Separate non-quantized and quantized models
        non_quantized_data = None
        quantized_data = []
        
        for model_name, data in model_data.items():
            if 'non_quantized' in model_name.lower():
                non_quantized_data = (model_name, data)
            else:
                quantized_data.append((model_name, data))
        
        # Sort quantized models by bit width
        def extract_bit_width(model_name):
            try:
                parts = model_name.split('_')
                for part in parts:
                    if 'bit' in part:
                        return int(part.replace('bit', ''))
                return 0
            except:
                return 0
        
        quantized_data.sort(key=lambda x: extract_bit_width(x[0]))
        
        # Add non-quantized model first
        if non_quantized_data:
            name, data = non_quantized_data
            val_acc = data['history'].get('val_accuracy', [])
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
        for model_name, data in quantized_data:
            val_acc = data['history'].get('val_accuracy', [])
            eval_results = data.get('eval_results', {})
            roc_auc = eval_results.get('roc_auc', None)
            n_trials = data.get('n_trials', 1)
            
            if val_acc and roc_auc is not None:
                display_name = self.format_model_name(model_name)
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
        ax1.set_title('Validation Accuracy and ROC AUC (Averaged)', fontsize=16, fontweight='bold')
        
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
        combined_plot_file = self.results_dir / "best_val_acc_and_roc_auc.png"
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Validation accuracy and ROC AUC chart saved to: {combined_plot_file}")
        
        return combined_plot_file
    
    def create_best_validation_accuracy_plot(self, model_data):
        """
        Create a square bar plot for best validation accuracy with custom styling.
        
        Args:
            model_data (dict): Dictionary containing model data
        """
        print("Creating best validation accuracy plot...")
        
        # Sort models: non-quantized first, then by weight bits
        sorted_models = []
        model_names = []
        
        for name, data in model_data.items():
            # Detect non-quantized models
            if any(keyword in name.lower() for keyword in ['non_quantized', 'float32', 'baseline']):
                sorted_models.append((name, data, 0))
                model_names.append('Float32')
            else:
                # Try to extract bit width from name
                bits = 999  # Default for unknown
                for possible_bits in [2, 3, 4, 6, 8, 16, 32]:
                    if f'{possible_bits}' in name:
                        bits = possible_bits
                        break
                sorted_models.append((name, data, bits))
                model_names.append(f'{bits}-bit')
        
        # Sort by weight bits
        sorted_indices = sorted(range(len(sorted_models)), key=lambda i: sorted_models[i][2])
        sorted_models = [sorted_models[i] for i in sorted_indices]
        model_names = [model_names[i] for i in sorted_indices]
        
        # Extract best validation accuracy
        best_val_acc = []
        
        for name, data, _ in sorted_models:
            if 'history' in data and 'val_accuracy' in data['history']:
                best_val_acc.append(max(data['history']['val_accuracy']))
            else:
                best_val_acc.append(0)  # Default if no data
        
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
        ax.set_title('Model2: Best Validation Accuracy', fontsize=28, fontweight='bold')  # Made smaller
        
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
        plot_file = self.results_dir / 'best_validation_accuracy_custom.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: best_validation_accuracy_custom.png")
        
        return plot_file
    
    def format_model_name(self, model_name):
        """
        Format model name for display in plots.
        
        Args:
            model_name (str): Original model directory name
            
        Returns:
            str: Formatted display name
        """
        if 'non_quantized' in model_name.lower():
            return 'Non-quantized'
        
        # Extract bit width and integer bits from model name
        try:
            # Handle various naming patterns
            if 'quantized_' in model_name:
                parts = model_name.replace('quantized_', '').replace('quantized_cnn_', '').split('_')
            else:
                parts = model_name.split('_')
            
            bit_part = None
            
            for part in parts:
                if 'bit' in part and bit_part is None:
                    bit_part = part
            
            if bit_part:
                bit_width = bit_part.replace('bit', '')
                return f'{bit_width}-bit'
            else:
                return model_name
        except:
            return model_name
    
    def generate_all_plots(self):
        """
        Generate all plots for the results directory.
        """
        print("=" * 70)
        print("GENERATING MODEL2 PLOTS WITH TRIAL AVERAGING")
        print("=" * 70)
        
        # Load model data
        model_data = self.load_model_data()
        
        if not model_data:
            print("No model data found in the results directory!")
            return
        
        # Create validation comparison plots
        acc_plot, loss_plot = self.create_validation_comparison_plots(model_data)
        
        # Create best metrics bar charts
        acc_bar, loss_bar = self.create_best_metrics_bar_charts(model_data)
        
        # Create validation accuracy and ROC AUC double y-axis chart
        combined_chart = self.create_val_acc_and_roc_auc_chart(model_data)
        
        # Create best validation accuracy chart
        validation_accuracy_chart = self.create_best_validation_accuracy_plot(model_data)
        
        print("\n" + "=" * 70)
        print("PLOT GENERATION COMPLETE!")
        print("=" * 70)
        print("Generated files:")
        print(f"  • {acc_plot.name}")
        print(f"  • {loss_plot.name}")
        print(f"  • {acc_bar.name}")
        print(f"  • {loss_bar.name}")
        if combined_chart:
            print(f"  • {combined_chart.name}")
        if validation_accuracy_chart:
            print(f"  • {validation_accuracy_chart.name}")
        
        # Check if ROC AUC plot was created
        auc_plot = self.results_dir / "roc_auc_comparison.png"
        if auc_plot.exists():
            print(f"  • {auc_plot.name}")
        
        print(f"\nAll plots saved to: {self.results_dir}")


def main():
    """
    Main function to parse arguments and generate plots.
    """
    parser = argparse.ArgumentParser(
        description="Generate plots from Model2 training results with trial averaging"
    )
    parser.add_argument(
        'results_dir',
        help='Path to the results directory (e.g., quantized_complicated_results_20250730_030326/)'
    )
    
    args = parser.parse_args()
    
    try:
        plotter = Model2ResultsPlotter(args.results_dir)
        plotter.generate_all_plots()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()