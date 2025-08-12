#!/usr/bin/env python3
"""
Standalone script to generate ONLY the best validation accuracy chart from Model2/Model3 training results.
Creates a single validation accuracy bar chart with custom styling.
Supports both Model1/Model2 format (overall_results.json + .npz) and Model3 format (individual JSON files).

Usage:
    python generate_validation_accuracy_chart.py <results_folder_path>
    
Examples:
    python generate_validation_accuracy_chart.py quantized_complicated_results_20250730_114020/
    python generate_validation_accuracy_chart.py combined_results_20250730_021907/
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
    Class to create ONLY the best validation accuracy chart from Model2/Model3 training results.
    Supports both data formats:
    - Model1/Model2: overall_results.json + .npz files (averaged data)
    - Model3: individual training_history.json + evaluation_results.json files
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
        
    def detect_data_format(self):
        """
        Detect whether this is Model1/Model2 format or Model3 format.
        
        Returns:
            str: 'model1_2' or 'model3'
        """
        # Look for a sample subdirectory to check format
        for subdir in self.results_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
                
            if not any(substring in subdir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            # Check for Model1/Model2 format (overall_results.json + .npz)
            if (subdir / "overall_results.json").exists():
                history_files = list(subdir.glob("*_history.npz"))
                if history_files:
                    print(f"Detected Model1/Model2 format (overall_results.json + .npz files)")
                    return 'model1_2'
            
            # Check for Model3 format (individual JSON files)
            if (subdir / "training_history.json").exists() and (subdir / "evaluation_results.json").exists():
                print(f"Detected Model3 format (individual JSON files)")
                return 'model3'
        
        raise ValueError("Could not detect data format. No valid model subdirectories found.")
        
    def load_model1_2_results_data(self):
        """
        Load data from Model1/Model2 results format (model subdirectories with overall_results.json + .npz).
        
        Returns:
            dict: Dictionary containing averaged model data
        """
        print("Loading Model1/Model2 results data...")
        
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
    
    def load_model3_results_data(self):
        """
        Load data from Model3 results format (individual JSON files).
        
        Returns:
            dict: Dictionary containing model data
        """
        print("Loading Model3 results data...")
        
        model_data = {}
        
        # Look for model subdirectories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            if not any(substring in model_dir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            print(f"  Processing: {model_dir.name}")
            
            # Load training history
            history_file = model_dir / "training_history.json"
            eval_file = model_dir / "evaluation_results.json"
            
            if not history_file.exists() or not eval_file.exists():
                print(f"    Warning: Missing required files in {model_dir.name}")
                continue
            
            try:
                # Load training history
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Load evaluation results
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                # Extract weight bits from directory name
                weight_bits = None
                if 'non_quantized' in model_dir.name:
                    model_type = 'non_quantized'
                    weight_bits = None
                else:
                    model_type = 'quantized'
                    # Extract bits from names like 'quantized_model_32bit_0int'
                    import re
                    bits_match = re.search(r'(\d+)bit', model_dir.name)
                    if bits_match:
                        weight_bits = int(bits_match.group(1))
                
                # Skip 32-bit models as requested
                if weight_bits == 32:
                    print(f"    Skipping 32-bit model: {model_dir.name}")
                    continue
                
                model_data[model_dir.name] = {
                    'history': history,
                    'eval_results': {
                        'roc_auc': eval_results.get('roc_auc'),
                        'test_accuracy': eval_results.get('test_accuracy'),
                        'test_loss': eval_results.get('test_loss')
                    },
                    'dir_name': model_dir.name,
                    'n_trials': 1,  # Model3 format is single trial
                    'weight_bits': weight_bits,
                    'model_type': model_type
                }
                
                print(f"    ✓ Loaded {len(history['val_accuracy'])} epochs of data")
                
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
        
        # Determine model number from directory path
        model_num = "Unknown"
        if "Model2" in str(self.results_dir):
            model_num = "Model2"
        elif "Model3" in str(self.results_dir):
            model_num = "Model3"
        elif "quantized_complicated_results" in str(self.results_dir):
            model_num = "Model2"
        elif "combined_results" in str(self.results_dir):
            model_num = "Model3"
        
        ax.set_title(f'{model_num}: Best Validation Accuracy', fontsize=28, fontweight='bold')  # Made smaller
        
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
        plot_file = self.results_dir / 'best_validation_accuracy.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Chart saved: {plot_file}")
        
        return plot_file
    
    def generate_chart(self):
        """
        Generate the validation accuracy chart.
        """
        print(f"\n=== Generating Validation Accuracy Chart ===")
        
        # Detect data format
        data_format = self.detect_data_format()
        
        # Load data based on format
        if data_format == 'model1_2':
            model_data = self.load_model1_2_results_data()
        elif data_format == 'model3':
            model_data = self.load_model3_results_data()
        else:
            raise ValueError(f"Unknown data format: {data_format}")
        
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
    parser = argparse.ArgumentParser(description='Generate best validation accuracy chart from Model2/Model3 training results')
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