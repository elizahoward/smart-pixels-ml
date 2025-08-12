#!/usr/bin/env python3
"""
Script to update the formatting of best_validation_accuracy_custom.png charts
to match the formatting of validation_accuracy_comparison.png charts.

This script will:
1. Use the same title, axis, and label sizes as validation_accuracy_comparison.png
2. Apply consistent formatting across Model1, Model2, and Model3
3. Regenerate the best_validation_accuracy_custom.png files

Usage:
    python update_custom_chart_formatting.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil


class CustomChartFormatter:
    """Updates custom chart formatting to match comparison charts"""
    
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir)
        self.custom_red = (139/255, 0/255, 33/255)
        
        # Define formatting to match validation_accuracy_comparison.png
        self.target_formatting = {
            'figsize': (8.5, 8.5),  # Square plot like comparison charts
            'title_fontsize': 16,   # Same as comparison charts
            'title_fontweight': 'bold',
            'ylabel_fontsize': 18,  # Same as comparison charts
            'xlabel_fontsize': 18,  # Same as comparison charts
            'tick_labelsize': 14,   # Same as comparison charts
            'legend_fontsize': 10,  # Same as comparison charts
            'bar_value_fontsize': 18,  # Larger, readable labels on bars
        }
        
    def find_results_directories(self):
        """Find all results directories that contain best_validation_accuracy_custom.png"""
        results_dirs = []
        
        for model_dir in ["Model1", "Model2", "Model3"]:
            model_path = self.base_dir / model_dir
            if not model_path.exists():
                print(f"Warning: {model_path} not found")
                continue
                
            for subdir in model_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    custom_chart = subdir / "best_validation_accuracy_custom.png"
                    if custom_chart.exists():
                        results_dirs.append(subdir)
                        print(f"Found: {subdir}")
        
        return results_dirs
    
    def detect_data_format(self, results_dir):
        """Detect whether this is Model1/Model2 format or Model3 format"""
        for subdir in results_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
                
            if not any(substring in subdir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            # Check for Model1/Model2 format (overall_results.json + .npz)
            if (subdir / "overall_results.json").exists():
                history_files = list(subdir.glob("*_history.npz"))
                if history_files:
                    return 'model1_2'
            
            # Check for Model3 format (individual JSON files)
            if (subdir / "training_history.json").exists() and (subdir / "evaluation_results.json").exists():
                return 'model3'
        
        raise ValueError("Could not detect data format")
    
    def load_model1_2_data(self, results_dir):
        """Load data from Model1/Model2 format"""
        model_data = {}
        
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            if not any(substring in model_dir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            overall_results_file = model_dir / "overall_results.json"
            if overall_results_file.exists():
                with open(overall_results_file, 'r') as f:
                    overall_results = json.load(f)
            else:
                overall_results = {}
            
            history_files = list(model_dir.glob("*_history.npz"))
            if not history_files:
                continue
            
            history_file = history_files[0]
            history_data = np.load(history_file)
            history = {
                'accuracy': history_data['accuracy'].tolist(),
                'val_accuracy': history_data['val_accuracy'].tolist(),
                'loss': history_data['loss'].tolist(),
                'val_loss': history_data['val_loss'].tolist()
            }
            
            # Skip 32-bit models
            if overall_results.get('weight_bits') == 32:
                continue
            
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
        
        return model_data
    
    def load_model3_data(self, results_dir):
        """Load data from Model3 format"""
        model_data = {}
        
        for model_dir in results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
                
            if not any(substring in model_dir.name for substring in ['quantized_', 'non_quantized']):
                continue
                
            history_file = model_dir / "training_history.json"
            eval_file = model_dir / "evaluation_results.json"
            
            if not history_file.exists() or not eval_file.exists():
                continue
            
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            with open(eval_file, 'r') as f:
                eval_results = json.load(f)
            
            # Extract weight bits from directory name
            weight_bits = None
            if 'non_quantized' in model_dir.name:
                model_type = 'non_quantized'
                weight_bits = None
            else:
                model_type = 'quantized'
                import re
                bits_match = re.search(r'(\d+)bit', model_dir.name)
                if bits_match:
                    weight_bits = int(bits_match.group(1))
            
            model_data[model_dir.name] = {
                'history': history,
                'eval_results': {
                    'roc_auc': eval_results.get('roc_auc'),
                    'test_accuracy': eval_results.get('test_accuracy'),
                    'test_loss': eval_results.get('test_loss')
                },
                'dir_name': model_dir.name,
                'n_trials': 1,
                'weight_bits': weight_bits,
                'model_type': model_type
            }
        
        return model_data
    
    def create_formatted_chart(self, model_data, results_dir):
        """Create best validation accuracy chart with comparison chart formatting"""
        print(f"Creating formatted chart for {results_dir.name}...")
        
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
        
        # Create figure with comparison chart formatting
        fig, ax = plt.subplots(figsize=self.target_formatting['figsize'])
        
        x_pos = np.arange(len(model_names))
        
        # Create colors list - grey for Float32, custom red for others
        colors = []
        for name in model_names:
            if name == 'Float32':
                colors.append('grey')
            else:
                colors.append(self.custom_red)
        
        # Create bars
        bars = ax.bar(x_pos, best_val_acc, color=colors, edgecolor='black', linewidth=1, width=0.6)
        
        # Determine model number from directory path
        model_num = "Unknown"
        if "Model1" in str(results_dir):
            model_num = "Model1"
        elif "Model2" in str(results_dir):
            model_num = "Model2"
        elif "Model3" in str(results_dir):
            model_num = "Model3"
        elif "quantized_complicated_results" in str(results_dir):
            model_num = "Model2"
        elif "combined_results" in str(results_dir):
            model_num = "Model3"
        elif "quantized_model1_results" in str(results_dir):
            model_num = "Model1"
        
        # Apply comparison chart formatting
        ax.set_title(f'{model_num}: Best Validation Accuracy', 
                    fontsize=self.target_formatting['title_fontsize'], 
                    fontweight=self.target_formatting['title_fontweight'])
        ax.set_ylabel('Best Validation Accuracy', 
                     fontsize=self.target_formatting['ylabel_fontsize'])
        
        # Set tick marks and labels with comparison chart formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.tick_params(axis='both', which='major', 
                      labelsize=self.target_formatting['tick_labelsize'])
        
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
        
        # Add value labels on top of bars with smaller font size
        for bar, val, color in zip(bars, best_val_acc, colors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontsize=self.target_formatting['bar_value_fontsize'], 
                    color=color)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_file = results_dir / 'best_validation_accuracy_custom.png'
        
        # Create backup of original
        backup_file = results_dir / 'best_validation_accuracy_custom_backup.png'
        if plot_file.exists() and not backup_file.exists():
            shutil.copy2(plot_file, backup_file)
            print(f"  Created backup: {backup_file.name}")
        
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Updated: {plot_file}")
        return plot_file
    
    def update_all_charts(self):
        """Update all custom charts to match comparison chart formatting"""
        print("=== Updating Custom Chart Formatting ===")
        print("Target formatting (matching validation_accuracy_comparison.png):")
        print(f"  - Figure size: {self.target_formatting['figsize']}")
        print(f"  - Title font size: {self.target_formatting['title_fontsize']}")
        print(f"  - Axis label font size: {self.target_formatting['ylabel_fontsize']}")
        print(f"  - Tick label size: {self.target_formatting['tick_labelsize']}")
        print()
        
        results_dirs = self.find_results_directories()
        
        if not results_dirs:
            print("❌ No results directories with custom charts found!")
            return
        
        updated_count = 0
        
        for results_dir in results_dirs:
            try:
                print(f"\n📁 Processing: {results_dir}")
                
                # Detect data format
                data_format = self.detect_data_format(results_dir)
                print(f"  Data format: {data_format}")
                
                # Load data
                if data_format == 'model1_2':
                    model_data = self.load_model1_2_data(results_dir)
                elif data_format == 'model3':
                    model_data = self.load_model3_data(results_dir)
                else:
                    print(f"  ❌ Unknown data format: {data_format}")
                    continue
                
                if not model_data:
                    print(f"  ❌ No model data found")
                    continue
                
                # Create formatted chart
                self.create_formatted_chart(model_data, results_dir)
                updated_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {results_dir}: {e}")
                continue
        
        print(f"\n✅ Successfully updated {updated_count} custom charts!")
        print("\n📝 Summary of changes:")
        print("  - Figure size: Changed from (9, 11) to (8.5, 8.5) - square plot")
        print("  - Title font size: Changed from 28 to 16")
        print("  - Y-axis label font size: Changed from 36 to 18")
        print("  - X-axis tick label size: Changed from 30 to 14")
        print("  - Y-axis tick label size: Changed from 20 to 14")
        print("  - Bar value font size: Changed to 18 (larger, readable labels)")
        print("\n💾 Original files backed up as *_backup.png")


def main():
    """Main function"""
    try:
        formatter = CustomChartFormatter()
        formatter.update_all_charts()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())