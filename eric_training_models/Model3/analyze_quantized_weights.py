#!/usr/bin/env python3
"""
True Quantized Weight Distribution Analysis

This script applies the actual quantization functions to the weights
to show the real quantized weight distributions, not just the 
fake-quantized weights stored during training.

Author: Eric
Date: Generated automatically
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Set matplotlib style
try:
    plt.style.use('ggplot')
except:
    plt.style.use('default')

class TrueQuantizedWeightAnalyzer:
    """
    Analyzes the actual quantized weight distributions by applying 
    quantization functions to the stored weights
    """
    
    def __init__(self, results_dir):
        """
        Initialize the analyzer with the results directory
        
        Args:
            results_dir (str): Path to the combined results directory
        """
        self.results_dir = Path(results_dir)
        self.models = {}
        self.model_parameters = {}
        
        # Define quantization bit levels
        self.quantization_levels = ['32bit', '16bit', '8bit', '6bit', '4bit', '3bit', '2bit']
        self.non_quantized_key = 'non_quantized'
        
    def quantize_weights(self, weights, bits, integer_bits=0):
        """
        Apply quantization to weights using the specified bit configuration
        
        Args:
            weights: Weight array to quantize
            bits: Number of bits for quantization
            integer_bits: Number of integer bits (default 0)
        
        Returns:
            Quantized weights
        """
        if bits >= 16:
            return weights  # No quantization for high precision
        
        # Calculate quantization parameters
        fractional_bits = bits - integer_bits
        
        # Calculate the scale factor
        max_val = 2**(integer_bits - 1) - 2**(-fractional_bits) if integer_bits > 0 else 1.0 - 2**(-fractional_bits)
        min_val = -2**(integer_bits - 1) if integer_bits > 0 else -1.0
        
        # Clip weights to quantization range
        clipped_weights = np.clip(weights, min_val, max_val)
        
        # Quantize
        scale = 2**fractional_bits
        quantized = np.round(clipped_weights * scale) / scale
        
        return quantized
    
    def load_model_weights(self):
        """Load weights from all models"""
        print("Loading model weights...")
        
        # Load non-quantized model
        non_quant_dir = self.results_dir / 'non_quantized_model'
        if non_quant_dir.exists():
            model_path = non_quant_dir / 'best_model.keras'
            if model_path.exists():
                print(f"Loading non-quantized model from {model_path}")
                try:
                    model = tf.keras.models.load_model(model_path)
                    self.models[self.non_quantized_key] = model
                    
                    # Load parameters
                    params_path = non_quant_dir / 'model_parameters.json'
                    if params_path.exists():
                        with open(params_path, 'r') as f:
                            self.model_parameters[self.non_quantized_key] = json.load(f)
                except Exception as e:
                    print(f"Error loading non-quantized model: {e}")
                    # Try loading from H5 as fallback
                    h5_path = non_quant_dir / 'best_model.h5'
                    if h5_path.exists():
                        print(f"Trying to load non-quantized model from H5: {h5_path}")
                        try:
                            weights = self._load_weights_from_h5(h5_path)
                            if weights:
                                self.models[self.non_quantized_key] = weights
                                # Load parameters
                                params_path = non_quant_dir / 'model_parameters.json'
                                if params_path.exists():
                                    with open(params_path, 'r') as f:
                                        self.model_parameters[self.non_quantized_key] = json.load(f)
                        except Exception as e2:
                            print(f"Error loading non-quantized model from H5: {e2}")
        
        # Load quantized model weights from H5 files
        for bits in self.quantization_levels:
            quant_dir = self.results_dir / f'quantized_model_{bits}_0int'
            if quant_dir.exists():
                h5_path = quant_dir / f'best_model_{bits}_0int.h5'
                if h5_path.exists():
                    print(f"Loading {bits} model weights from {h5_path}")
                    try:
                        weights = self._load_weights_from_h5(h5_path)
                        if weights:
                            self.models[bits] = weights
                            
                            # Load parameters
                            params_path = quant_dir / 'model_parameters.json'
                            if params_path.exists():
                                with open(params_path, 'r') as f:
                                    self.model_parameters[bits] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {bits} model: {e}")
        
        print(f"Successfully loaded weights from {len(self.models)} models")
        return len(self.models) > 0
    
    def _load_weights_from_h5(self, h5_path):
        """Load weights from H5 file"""
        try:
            import h5py
            weights_dict = {}
            
            with h5py.File(h5_path, 'r') as f:
                def extract_weights(group, prefix=""):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            extract_weights(item, f"{prefix}{key}_")
                        elif isinstance(item, h5py.Dataset):
                            weights_dict[f"{prefix}{key}"] = item[:]
                
                if 'model_weights' in f:
                    extract_weights(f['model_weights'])
                else:
                    extract_weights(f)
            
            return weights_dict
        except Exception as e:
            print(f"Error loading weights from H5: {e}")
            return None
    
    def analyze_quantization_effects(self):
        """Analyze the true quantization effects by applying quantization"""
        
        print("Analyzing true quantization effects...")
        
        # Create output directory
        output_dir = self.results_dir / 'true_quantized_analysis'
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for analysis
        analysis_data = {}
        
        for model_name, model_data in self.models.items():
            print(f"Processing {model_name}...")
            
            if model_name == self.non_quantized_key:
                # Extract weights from Keras model or weights dict
                if hasattr(model_data, 'layers'):
                    # It's a Keras model
                    weights_dict = {}
                    for i, layer in enumerate(model_data.layers):
                        layer_weights = layer.get_weights()
                        if len(layer_weights) > 0:
                            weights_dict[f"layer_{i}_kernel"] = layer_weights[0]
                            if len(layer_weights) > 1:
                                weights_dict[f"layer_{i}_bias"] = layer_weights[1]
                else:
                    # It's already a weights dictionary
                    weights_dict = model_data
                
                analysis_data[model_name] = {
                    'original_weights': weights_dict,
                    'quantized_weights': weights_dict,  # No quantization
                    'params': self.model_parameters.get(model_name, {})
                }
            else:
                # Apply quantization to loaded weights
                if model_name in self.model_parameters:
                    params = self.model_parameters[model_name]
                    weight_bits = params.get('weight_bits', 8)
                    integer_bits = params.get('integer_bits', 0)
                    
                    quantized_weights = {}
                    for weight_name, weight_array in model_data.items():
                        quantized_weights[weight_name] = self.quantize_weights(
                            weight_array, weight_bits, integer_bits
                        )
                    
                    analysis_data[model_name] = {
                        'original_weights': model_data,
                        'quantized_weights': quantized_weights,
                        'params': params
                    }
        
        # Generate comparison plots
        self._create_quantization_comparison_plots(analysis_data, output_dir)
        self._create_unique_values_analysis(analysis_data, output_dir)
        self._create_distribution_comparison(analysis_data, output_dir)
        self._create_overlay_quantization_plot(analysis_data, output_dir)
        self._create_standalone_unique_values_plot(analysis_data, output_dir)
        self._create_zoomed_quantization_plot(analysis_data, output_dir)
        self._generate_quantization_report(analysis_data, output_dir)
        
        print(f"Analysis complete! Results saved to {output_dir}")
    
    def _create_quantization_comparison_plots(self, analysis_data, output_dir):
        """Create before/after quantization comparison plots"""
        
        fig, axes = plt.subplots(2, 4, figsize=(25, 10))
        fig.suptitle('Weight Distributions: Original vs Quantized', fontsize=28)
        
        model_names = list(analysis_data.keys())
        
        for i, model_name in enumerate(model_names[:4]):  # Show first 4 models
            if model_name == self.non_quantized_key:
                continue
                
            data = analysis_data[model_name]
            
            # Collect all weights
            original_weights = []
            quantized_weights = []
            
            for weight_name, weight_array in data['original_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    original_weights.extend(weight_array.flatten())
                    quantized_weights.extend(data['quantized_weights'][weight_name].flatten())
            
            # Plot original weights
            axes[0, i].hist(original_weights, bins=50, alpha=0.7, color='blue', density=True)
            axes[0, i].set_title(f'{model_name} - Original Weights', fontsize=20)
            axes[0, i].set_xlabel('Weight Value', fontsize=18)
            axes[0, i].set_ylabel('Density', fontsize=18)
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot quantized weights
            axes[1, i].hist(quantized_weights, bins=50, alpha=0.7, color='red', density=True)
            axes[1, i].set_title(f'{model_name} - Quantized Weights', fontsize=20)
            axes[1, i].set_xlabel('Weight Value', fontsize=18)
            axes[1, i].set_ylabel('Density', fontsize=18)
            axes[1, i].grid(True, alpha=0.3)
            
            # Add unique values count
            unique_original = len(np.unique(original_weights))
            unique_quantized = len(np.unique(quantized_weights))
            
            axes[0, i].text(0.7, 0.9, f'Unique: {unique_original:,}', 
                           transform=axes[0, i].transAxes, fontsize=16, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            axes[1, i].text(0.7, 0.9, f'Unique: {unique_quantized:,}', 
                           transform=axes[1, i].transAxes, fontsize=16, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quantization_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_unique_values_analysis(self, analysis_data, output_dir):
        """Analyze unique values before and after quantization"""
        
        fig, axes = plt.subplots(1, 3, figsize=(22.5, 6))
        fig.suptitle('Quantization Effects: Unique Values Analysis', fontsize=28)
        
        models = []
        original_unique = []
        quantized_unique = []
        theoretical_max = []
        
        for model_name, data in analysis_data.items():
            if model_name == self.non_quantized_key:
                continue
                
            # Collect all weights
            orig_weights = []
            quant_weights = []
            
            for weight_name, weight_array in data['original_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    orig_weights.extend(weight_array.flatten())
                    quant_weights.extend(data['quantized_weights'][weight_name].flatten())
            
            models.append(model_name)
            original_unique.append(len(np.unique(orig_weights)))
            quantized_unique.append(len(np.unique(quant_weights)))
            
            # Calculate theoretical maximum
            weight_bits = data['params'].get('weight_bits', 8)
            theoretical_max.append(2**weight_bits)
        
        # Plot 1: Unique values comparison
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, original_unique, width, label='Original', alpha=0.7)
        axes[0].bar(x + width/2, quantized_unique, width, label='Quantized', alpha=0.7)
        axes[0].set_title('Unique Values: Original vs Quantized', fontsize=20)
        axes[0].set_xlabel('Model', fontsize=18)
        axes[0].set_ylabel('Number of Unique Values', fontsize=18)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (orig, quant) in enumerate(zip(original_unique, quantized_unique)):
            axes[0].text(i - width/2, orig, f'{orig:,}', ha='center', va='bottom', fontsize=15)
            axes[0].text(i + width/2, quant, f'{quant:,}', ha='center', va='bottom', fontsize=15)
        
        # Plot 2: Compression ratio
        compression_ratios = [orig/quant for orig, quant in zip(original_unique, quantized_unique)]
        
        bars = axes[1].bar(models, compression_ratios, alpha=0.7, color='green')
        axes[1].set_title('Compression Ratio (Original/Quantized)', fontsize=20)
        axes[1].set_xlabel('Model', fontsize=18)
        axes[1].set_ylabel('Compression Ratio', fontsize=18)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, compression_ratios):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{ratio:.1f}x', ha='center', va='bottom', fontsize=15)
        
        # Plot 3: Theoretical vs Actual
        axes[2].bar(x - width/2, theoretical_max, width, label='Theoretical Max', alpha=0.7)
        axes[2].bar(x + width/2, quantized_unique, width, label='Actual Quantized', alpha=0.7)
        axes[2].set_title('Theoretical vs Actual Unique Values', fontsize=20)
        axes[2].set_xlabel('Model', fontsize=18)
        axes[2].set_ylabel('Number of Unique Values', fontsize=18)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'unique_values_true_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_distribution_comparison(self, analysis_data, output_dir):
        """Create distribution comparison plots"""
        
        n_models = len([k for k in analysis_data.keys() if k != self.non_quantized_key])
        fig, axes = plt.subplots(1, min(n_models, 4), figsize=(20, 4))
        if n_models == 1:
            axes = [axes]
        fig.suptitle('Quantized Weight Distributions by Bit Level', fontsize=28)
        
        plot_idx = 0
        for model_name, data in analysis_data.items():
            if model_name == self.non_quantized_key or plot_idx >= 4:
                continue
                
            # Collect quantized weights
            quant_weights = []
            for weight_name, weight_array in data['quantized_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    quant_weights.extend(weight_array.flatten())
            
            unique_vals = np.unique(quant_weights)
            weight_bits = data['params'].get('weight_bits', 8)
            
            axes[plot_idx].hist(quant_weights, bins=min(50, len(unique_vals)), alpha=0.7, density=True)
            axes[plot_idx].set_title(f'{weight_bits}-bit Quantized\n{len(unique_vals)} unique values', fontsize=20)
            axes[plot_idx].set_xlabel('Weight Value', fontsize=18)
            axes[plot_idx].set_ylabel('Density', fontsize=18)
            axes[plot_idx].grid(True, alpha=0.3)
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quantized_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_overlay_quantization_plot(self, analysis_data, output_dir):
        """Create overlay plot showing all quantization levels with fine bins to show discrete values"""
        
        # Set style for white background
        plt.style.use('default')
        
        fig, axes = plt.subplots(1, 2, figsize=(25, 8), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')
        fig.suptitle('Quantized Weight Distributions Overlay - Discrete Quantization Levels', fontsize=28, color='black')
        
        # Define colors for different bit levels
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink']
        
        # Collect data for all quantized models
        all_weights_data = {}
        weight_ranges = []
        
        for model_name, data in analysis_data.items():
            if model_name == self.non_quantized_key:
                continue
                
            # Collect quantized weights
            quant_weights = []
            for weight_name, weight_array in data['quantized_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    quant_weights.extend(weight_array.flatten())
            
            if quant_weights:
                all_weights_data[model_name] = np.array(quant_weights)
                weight_ranges.extend([np.min(quant_weights), np.max(quant_weights)])
        
        if not all_weights_data:
            print("No quantized weight data found for overlay plot")
            return
        
        # Calculate overall range for consistent binning
        overall_min = min(weight_ranges)
        overall_max = max(weight_ranges)
        
        # Create very fine bins to show discrete quantization levels
        # Use many bins to capture the discrete nature
        n_bins = 2000  # Very fine bins
        bin_edges = np.linspace(overall_min, overall_max, n_bins + 1)
        
        # Plot 1: Full range overlay
        for i, (model_name, weights) in enumerate(all_weights_data.items()):
            color = colors[i % len(colors)]
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            unique_count = len(np.unique(weights))
            
            # Create histogram with very fine bins
            counts, _ = np.histogram(weights, bins=bin_edges, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Plot as a line to better show the discrete levels
            axes[0].plot(bin_centers, counts, 
                        label=f'{weight_bits}-bit ({unique_count:,} unique)', 
                        color=color, linewidth=1.5, alpha=0.8)
            
            # Also add a scatter plot to emphasize discrete points
            # Sample some points to avoid overcrowding
            sample_indices = counts > 0
            if np.any(sample_indices):
                axes[0].scatter(bin_centers[sample_indices], counts[sample_indices], 
                              color=color, s=0.5, alpha=0.6)
        
        axes[0].set_title('All Quantization Levels - Full Range', fontsize=20)
        axes[0].set_xlabel('Weight Value', fontsize=18)
        axes[0].set_ylabel('Density (Log Scale)', fontsize=18)
        axes[0].set_yscale('log')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Zoomed in view around zero to better see quantization steps
        zoom_range = 0.2  # Zoom to ±0.2 around zero
        zoom_mask = (bin_centers >= -zoom_range) & (bin_centers <= zoom_range)
        
        for i, (model_name, weights) in enumerate(all_weights_data.items()):
            color = colors[i % len(colors)]
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            
            # Filter weights in zoom range
            zoom_weights = weights[(weights >= -zoom_range) & (weights <= zoom_range)]
            if len(zoom_weights) > 0:
                unique_count_zoom = len(np.unique(zoom_weights))
                
                # Create histogram for zoom range
                zoom_bins = np.linspace(-zoom_range, zoom_range, 500)
                counts, _ = np.histogram(zoom_weights, bins=zoom_bins, density=True)
                bin_centers_zoom = (zoom_bins[:-1] + zoom_bins[1:]) / 2
                
                # Plot as both line and scatter
                axes[1].plot(bin_centers_zoom, counts, 
                            label=f'{weight_bits}-bit ({unique_count_zoom} in range)', 
                            color=color, linewidth=2, alpha=0.8)
                
                # Add markers for discrete levels
                discrete_mask = counts > 0
                if np.any(discrete_mask):
                    axes[1].scatter(bin_centers_zoom[discrete_mask], counts[discrete_mask], 
                                  color=color, s=3, alpha=0.8, marker='o')
        
        axes[1].set_title(f'Quantization Levels - Zoomed View (±{zoom_range})', fontsize=20)
        axes[1].set_xlabel('Weight Value', fontsize=18)
        axes[1].set_ylabel('Density (Log Scale)', fontsize=18)
        axes[1].set_yscale('log')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overlay_quantization_discrete_levels.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Create a second figure showing just the unique values as discrete points
        self._create_discrete_values_plot(all_weights_data, analysis_data, output_dir)
    
    def _create_discrete_values_plot(self, all_weights_data, analysis_data, output_dir):
        """Create a plot showing the actual discrete quantization values"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Discrete Quantization Values - Actual Quantization Levels', fontsize=28)
        
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink']
        
        # Plot 1: Unique values scatter plot
        y_offset = 0
        for i, (model_name, weights) in enumerate(all_weights_data.items()):
            color = colors[i % len(colors)]
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            
            unique_values = np.unique(weights)
            # Limit to reasonable number for visualization
            if len(unique_values) > 200:
                # Sample unique values for visualization
                indices = np.linspace(0, len(unique_values)-1, 200, dtype=int)
                unique_values = unique_values[indices]
            
            y_positions = np.full_like(unique_values, y_offset)
            axes[0, 0].scatter(unique_values, y_positions, 
                             color=color, alpha=0.7, s=20, 
                             label=f'{weight_bits}-bit')
            y_offset += 1
        
        axes[0, 0].set_title('Unique Quantized Values Distribution', fontsize=20)
        axes[0, 0].set_xlabel('Weight Value', fontsize=18)
        axes[0, 0].set_ylabel('Quantization Level', fontsize=18)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Quantization step sizes
        step_sizes = []
        bit_levels = []
        
        for model_name, weights in all_weights_data.items():
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            unique_values = np.unique(weights)
            
            if len(unique_values) > 1:
                # Calculate average step size
                diffs = np.diff(sorted(unique_values))
                # Remove zero differences and very small differences (noise)
                diffs = diffs[diffs > 1e-10]
                if len(diffs) > 0:
                    avg_step = np.mean(diffs)
                    step_sizes.append(avg_step)
                    bit_levels.append(weight_bits)
        
        if step_sizes:
            axes[0, 1].bar([str(b) for b in bit_levels], step_sizes, 
                          color=colors[:len(step_sizes)], alpha=0.7)
            axes[0, 1].set_title('Average Quantization Step Size', fontsize=20)
            axes[0, 1].set_xlabel('Bit Level', fontsize=18)
            axes[0, 1].set_ylabel('Average Step Size', fontsize=18)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')
        
        # Plot 3: Histogram of unique values count
        unique_counts = []
        bit_labels = []
        
        for model_name, weights in all_weights_data.items():
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            unique_count = len(np.unique(weights))
            unique_counts.append(unique_count)
            bit_labels.append(f'{weight_bits}-bit')
        
        bars = axes[1, 0].bar(bit_labels, unique_counts, 
                             color=colors[:len(unique_counts)], alpha=0.7)
        axes[1, 0].set_title('Number of Unique Values per Quantization Level', fontsize=20)
        axes[1, 0].set_xlabel('Quantization Level', fontsize=18)
        axes[1, 0].set_ylabel('Number of Unique Values', fontsize=18)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, unique_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count:,}', ha='center', va='bottom', fontsize=16)
        
        # Plot 4: Theoretical vs Actual quantization levels
        theoretical_vals = []
        actual_vals = []
        bit_levels_theory = []
        
        for model_name, weights in all_weights_data.items():
            weight_bits = analysis_data[model_name]['params'].get('weight_bits', 8)
            theoretical = 2**weight_bits
            actual = len(np.unique(weights))
            
            theoretical_vals.append(theoretical)
            actual_vals.append(actual)
            bit_levels_theory.append(weight_bits)
        
        x = np.arange(len(bit_levels_theory))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, theoretical_vals, width, 
                      label='Theoretical', alpha=0.7, color='lightblue')
        axes[1, 1].bar(x + width/2, actual_vals, width, 
                      label='Actual', alpha=0.7, color='darkblue')
        
        axes[1, 1].set_title('Theoretical vs Actual Unique Values', fontsize=20)
        axes[1, 1].set_xlabel('Bit Level', fontsize=18)
        axes[1, 1].set_ylabel('Number of Unique Values', fontsize=18)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'{b}-bit' for b in bit_levels_theory])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'discrete_quantization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_standalone_unique_values_plot(self, analysis_data, output_dir):
        """Create a standalone unique quantized values distribution plot"""
        
        # Set style for white background
        plt.style.use('default')
        
        # Create more square figure with white background
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Define colors for different bit levels
        colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']
        
        bit_levels = []
        y_positions = []
        y_labels = []
        
        # Start with non-quantized if available
        y_offset = 0
        if self.non_quantized_key in analysis_data:
            # Get non-quantized weights
            non_quant_data = analysis_data[self.non_quantized_key]
            non_quant_weights = []
            for weight_name, weight_array in non_quant_data['original_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    non_quant_weights.extend(weight_array.flatten())
            
            if non_quant_weights:
                unique_values = np.unique(non_quant_weights)
                # Sample for visualization if too many
                if len(unique_values) > 500:
                    indices = np.linspace(0, len(unique_values)-1, 500, dtype=int)
                    unique_values = unique_values[indices]
                
                y_positions_array = np.full_like(unique_values, y_offset)
                ax.scatter(unique_values, y_positions_array, 
                          color='black', alpha=0.7, s=15, 
                          label='Non-quantized', marker='o')
                
                bit_levels.append('Non-quantized')
                y_positions.append(y_offset)
                y_labels.append('Non-quantized')
                y_offset += 1
        
        # Add quantized models
        # Sort by bit level for better visualization
        quantized_models = [(k, v) for k, v in analysis_data.items() if k != self.non_quantized_key]
        quantized_models.sort(key=lambda x: x[1]['params'].get('weight_bits', 32), reverse=True)
        
        for i, (model_name, data) in enumerate(quantized_models):
            color = colors[i % len(colors)]
            weight_bits = data['params'].get('weight_bits', 8)
            
            # Get quantized weights
            quant_weights = []
            for weight_name, weight_array in data['quantized_weights'].items():
                if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                    quant_weights.extend(weight_array.flatten())
            
            if quant_weights:
                unique_values = np.unique(quant_weights)
                unique_count = len(unique_values)
                
                # Sample for visualization if too many
                if len(unique_values) > 200:
                    indices = np.linspace(0, len(unique_values)-1, 200, dtype=int)
                    unique_values = unique_values[indices]
                
                y_positions_array = np.full_like(unique_values, y_offset)
                ax.scatter(unique_values, y_positions_array, 
                          color=color, alpha=0.8, s=20, 
                          label=f'{weight_bits}-bit ({unique_count:,} unique)', 
                          marker='o')
                
                bit_levels.append(f'{weight_bits}-bit')
                y_positions.append(y_offset)
                y_labels.append(f'{weight_bits}-bit')
                y_offset += 1
        
        # Customize the plot
        ax.set_title('Unique Quantized Values Distribution', fontsize=26, fontweight='bold', pad=20)
        ax.set_xlabel('Weight Value', fontsize=20, fontweight='bold')
        ax.set_ylabel('Quantization Level', fontsize=20, fontweight='bold')
        
        # Set y-axis to show bit labels instead of numbers
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=18)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # No legend for cleaner look
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=17, 
                      colors='black', width=1.5, length=6)
        
        # Make it more square and compact
        ax.set_aspect('auto')
        
        # Set reasonable limits
        if len(bit_levels) > 0:
            ax.set_ylim(-0.5, len(bit_levels) - 0.5)
        
        # Make the plot more compact and square
        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
        
        # Save with high DPI and tight bbox
        plt.tight_layout()
        plt.savefig(output_dir / 'standalone_unique_values_distribution.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Standalone unique values plot saved to {output_dir / 'standalone_unique_values_distribution.png'}")
    
    def _create_zoomed_quantization_plot(self, analysis_data, output_dir):
        """Create a standalone zoomed quantization plot with original color scheme"""
        
        # Set style for white background
        plt.style.use('default')
        
        # Create square figure 
        fig, ax = plt.subplots(1, 1, figsize=(12.5, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Define pastel color scheme
        pastel_colors = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#D4BAFF', '#FFBAF3', '#F0F0F0']
        # Pastel red, peach, light yellow, mint green, light blue, lavender, light pink, light gray
        
        # Collect data for all models including non-quantized
        all_weights_data = {}
        weight_ranges = []
        
        # Process all models (quantized + non-quantized)
        for model_name, data in analysis_data.items():
            model_weights = []
            
            if model_name == self.non_quantized_key:
                # Get non-quantized weights
                for weight_name, weight_array in data['original_weights'].items():
                    if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                        model_weights.extend(weight_array.flatten())
            else:
                # Get quantized weights
                for weight_name, weight_array in data['quantized_weights'].items():
                    if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                        model_weights.extend(weight_array.flatten())
            
            if model_weights:
                all_weights_data[model_name] = np.array(model_weights)
                weight_ranges.extend([np.min(model_weights), np.max(model_weights)])
        
        if not all_weights_data:
            print("No weight data found for zoomed quantization plot")
            return
        
        # Zoomed view around zero
        zoom_range = 0.2  # ±0.2 around zero
        
        # Sort models for plotting (quantized first, non-quantized last for foreground)
        ordered_models = []
        
        # Add quantized models sorted by bit level (highest to lowest)
        quantized_models = [(k, v) for k, v in analysis_data.items() if k != self.non_quantized_key]
        quantized_models.sort(key=lambda x: x[1]['params'].get('weight_bits', 32), reverse=True)
        for model_name, data in quantized_models:
            if model_name in all_weights_data:
                weight_bits = data['params'].get('weight_bits', 8)
                ordered_models.append((model_name, f'{weight_bits}-bit'))
        
        # Add non-quantized last so it appears on top
        if self.non_quantized_key in all_weights_data:
            ordered_models.append((self.non_quantized_key, 'Non-quantized'))
        
        # Plot each model with specific color mapping
        for i, (model_name, display_name) in enumerate(ordered_models):
            weights = all_weights_data[model_name]
            
            # Assign specific colors for better visualization
            if model_name == self.non_quantized_key:
                color = 'black'  # Black for non-quantized (foreground)
            elif '8bit' in model_name:
                color = '#FFB3BA'  # Pastel red for 8-bit
            elif '2bit' in model_name:
                color = '#FFBAF3'  # Light pink for 2-bit
            elif '3bit' in model_name:
                color = '#D4BAFF'  # Lavender for 3-bit
            elif '4bit' in model_name:
                color = '#BAE1FF'  # Light blue for 4-bit
            elif '6bit' in model_name:
                color = '#BAFFC9'  # Mint green for 6-bit
            elif '16bit' in model_name:
                color = '#FFFFBA'  # Light yellow for 16-bit
            elif '32bit' in model_name:
                color = '#FFDFBA'  # Peach for 32-bit
            else:
                color = pastel_colors[i % len(pastel_colors)]
            
            # Filter weights in zoom range
            zoom_weights = weights[(weights >= -zoom_range) & (weights <= zoom_range)]
            if len(zoom_weights) > 0:
                unique_count_zoom = len(np.unique(zoom_weights))
                
                # Create histogram for zoom range with fine bins
                zoom_bins = np.linspace(-zoom_range, zoom_range, 800)  # Fine bins for detail
                counts, _ = np.histogram(zoom_weights, bins=zoom_bins, density=True)
                bin_centers_zoom = (zoom_bins[:-1] + zoom_bins[1:]) / 2
                
                # Plot as line for smooth appearance - make non-quantized thicker
                linewidth = 4.0 if model_name == self.non_quantized_key else 3.0
                alpha = 1.0 if model_name == self.non_quantized_key else 0.9
                
                ax.plot(bin_centers_zoom, counts, 
                       label=f'{display_name} ({unique_count_zoom} unique)', 
                       color=color, linewidth=linewidth, alpha=alpha)
                
                # Add scatter points for discrete levels where density > 0
                discrete_mask = counts > 0
                if np.any(discrete_mask):
                    scatter_size = 8 if model_name == self.non_quantized_key else 6
                    scatter_alpha = 1.0 if model_name == self.non_quantized_key else 0.8
                    ax.scatter(bin_centers_zoom[discrete_mask], counts[discrete_mask], 
                             color=color, s=scatter_size, alpha=scatter_alpha, marker='o')
        
        # Customize the plot
        ax.set_title('Quantisation Weights - Model3', fontsize=26, fontweight='bold', pad=20, color='black')
        ax.set_xlabel('Weight Value', fontsize=20, fontweight='bold')
        ax.set_ylabel('Density (Log Scale)', fontsize=20, fontweight='bold')
        ax.set_yscale('log')
        
        # Set x-axis limits to zoom range
        ax.set_xlim(-zoom_range, zoom_range)
        
        # Style the plot
        ax.legend(loc='upper right', fontsize=16, frameon=True, fancybox=True, shadow=True)
        ax.legend().get_frame().set_facecolor('white')
        ax.legend().get_frame().set_alpha(0.95)
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=18, 
                      colors='black', width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', labelsize=16, 
                      colors='black', width=1, length=4)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_dir / 'zoomed_quantization_weights_model3.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Zoomed quantization plot saved to {output_dir / 'zoomed_quantization_weights_model3.png'}")
    
    def _generate_quantization_report(self, analysis_data, output_dir):
        """Generate a detailed quantization analysis report"""
        
        report_path = output_dir / 'true_quantization_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("TRUE QUANTIZATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("This analysis applies actual quantization to the stored weights\n")
            f.write("to show the real quantized weight distributions.\n\n")
            
            f.write("QUANTIZATION EFFECTS (ACTUAL):\n")
            f.write("-" * 35 + "\n")
            
            # Reference model
            if self.non_quantized_key in analysis_data:
                ref_data = analysis_data[self.non_quantized_key]
                ref_weights = []
                for weight_name, weight_array in ref_data['original_weights'].items():
                    if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                        ref_weights.extend(weight_array.flatten())
                
                ref_unique = len(np.unique(ref_weights))
                f.write(f"Original Model Unique Values: {ref_unique:,}\n\n")
            
            for model_name, data in analysis_data.items():
                if model_name == self.non_quantized_key:
                    continue
                    
                # Collect quantized weights
                quant_weights = []
                for weight_name, weight_array in data['quantized_weights'].items():
                    if 'kernel' in weight_name.lower() or 'weight' in weight_name.lower():
                        quant_weights.extend(weight_array.flatten())
                
                quant_unique = len(np.unique(quant_weights))
                weight_bits = data['params'].get('weight_bits', 8)
                theoretical_max = 2**weight_bits
                
                if self.non_quantized_key in analysis_data:
                    compression_ratio = ref_unique / quant_unique
                    size_reduction = (1 - quant_unique/ref_unique) * 100
                else:
                    compression_ratio = 1.0
                    size_reduction = 0.0
                
                f.write(f"{model_name}:\n")
                f.write(f"  Weight Bits: {weight_bits}\n")
                f.write(f"  Theoretical Max Unique Values: {theoretical_max:,}\n")
                f.write(f"  Actual Unique Values: {quant_unique:,}\n")
                f.write(f"  Compression Ratio: {compression_ratio:.2f}x\n")
                f.write(f"  Size Reduction: {size_reduction:.1f}%\n")
                f.write(f"  Utilization: {(quant_unique/theoretical_max)*100:.1f}% of theoretical maximum\n\n")
        
        print(f"True quantization report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete true quantization analysis"""
        
        print("Starting True Quantized Weight Analysis...")
        print("=" * 50)
        
        if not self.load_model_weights():
            print("Error: Could not load model weights!")
            return False
        
        self.analyze_quantization_effects()
        
        print("\n" + "=" * 50)
        print("True quantization analysis complete!")
        return True


def main():
    """Main function"""
    
    results_dir = "combined_results_20250730_021907"
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
        return
    
    analyzer = TrueQuantizedWeightAnalyzer(results_dir)
    
    try:
        success = analyzer.run_analysis()
        if success:
            print("\n✅ True quantization analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()