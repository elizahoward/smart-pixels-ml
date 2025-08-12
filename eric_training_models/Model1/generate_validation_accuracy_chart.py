#!/usr/bin/env python3
"""
Standalone script to generate ONLY the best validation accuracy chart from Model1 training results.
Creates a single validation accuracy bar chart with custom styling.

Usage:
    python generate_validation_accuracy_chart.py <results_folder_path>
    
Examples:
    python generate_validation_accuracy_chart.py quantized_model1_results_20250731_134522/
    python generate_validation_accuracy_chart.py .  # To process current directory
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import argparse

class ValidationAccuracyChartGenerator:
    """
    Class to create ONLY the best validation accuracy chart from Model1 training results.
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
        
        # Custom red color (139, 0, 33) for histogram bars
        self.custom_red = (139/255, 0/255, 33/255)
        
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
    
    def create_best_validation_accuracy_chart(self, model_data):
        """
        Create a square bar plot for best validation accuracy with custom styling.
        
        Args:
            model_data (dict): Dictionary containing model data
        """
        print("Creating best validation accuracy chart...")
        
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
        ax.set_ylim(0.6, 1.0)
        
        # Add horizontal gridlines
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis tick labels (remove the 1.00 tick)
        yticks = ax.get_yticks()
        filtered_ticks = [tick for tick in yticks if abs(tick - 1.0) > 0.01]
        ax.set_yticks(filtered_ticks)
        yticklabels = [f'{tick:.2f}' for tick in filtered_ticks]
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
        
        print(f"✅ Chart saved: {plot_file}")
        
        return plot_file
    
    def generate_chart(self):
        """
        Generate the validation accuracy chart.
        """
        print(f"\n=== Generating Model1 Validation Accuracy Chart ===")
        
        # Load data
        model_data = self.load_model1_results_data()
        
        if not model_data:
            print("❌ No model data found!")
            return
        
        print(f"\nFound {len(model_data)} model configurations:")
        for name in model_data.keys():
            print(f"  - {name}")
        
        # Create chart
        chart_file = self.create_best_validation_accuracy_chart(model_data)
        
        print(f"\n✅ Validation accuracy chart generated successfully!")
        print(f"Saved as: {chart_file}")

def main():
    """Main function to run the chart generation script"""
    parser = argparse.ArgumentParser(description='Generate best validation accuracy chart from Model1 training results')
    parser.add_argument('results_dir', help='Path to results directory')
    
    args = parser.parse_args()
    
    try:
        generator = ValidationAccuracyChartGenerator(args.results_dir)
        generator.generate_chart()
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()