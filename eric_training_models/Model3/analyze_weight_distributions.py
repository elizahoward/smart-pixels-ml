#!/usr/bin/env python3
"""
Weight Distribution Analysis for Quantized Models

This script analyzes the weight and bias distributions of quantized models
compared to the non-quantized baseline model. It extracts weights from 
Keras model files and generates comprehensive visualizations showing 
the effects of quantization on different layers.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Seaborn not available - using matplotlib defaults")
    SEABORN_AVAILABLE = False

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("SciPy not available - skipping advanced statistical analysis")
    SCIPY_AVAILABLE = False
from datetime import datetime

# Set matplotlib style for better plots
try:
    if SEABORN_AVAILABLE:
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    else:
        plt.style.use('ggplot')
except:
    plt.style.use('default')
    print("Using default matplotlib style")

class WeightDistributionAnalyzer:
    """
    Analyzes weight distributions across different quantization levels
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
        
        # Define quantization bit levels to analyze
        self.quantization_levels = ['32bit', '16bit', '8bit', '6bit', '4bit', '3bit', '2bit']
        self.non_quantized_key = 'non_quantized'
        
    def load_models(self):
        """
        Load all available models from the results directory
        """
        print("Loading models...")
        
        # Load non-quantized model
        non_quant_dir = self.results_dir / 'non_quantized_model'
        if non_quant_dir.exists():
            model_path = non_quant_dir / 'best_model.keras'
            if model_path.exists():
                print(f"Loading non-quantized model from {model_path}")
                self.models[self.non_quantized_key] = tf.keras.models.load_model(model_path)
                
                # Load parameters
                params_path = non_quant_dir / 'model_parameters.json'
                if params_path.exists():
                    with open(params_path, 'r') as f:
                        self.model_parameters[self.non_quantized_key] = json.load(f)
        
        # Load quantized models
        for bits in self.quantization_levels:
            quant_dir = self.results_dir / f'quantized_model_{bits}_0int'
            if quant_dir.exists():
                model_path = quant_dir / f'best_model_{bits}_0int.keras'
                if model_path.exists():
                    print(f"Loading {bits} quantized model from {model_path}")
                    try:
                        # Load with custom objects for QKeras layers
                        custom_objects = self._get_qkeras_custom_objects()
                        self.models[bits] = tf.keras.models.load_model(
                            model_path, 
                            custom_objects=custom_objects
                        )
                        
                        # Load parameters
                        params_path = quant_dir / 'model_parameters.json'
                        if params_path.exists():
                            with open(params_path, 'r') as f:
                                self.model_parameters[bits] = json.load(f)
                                
                    except Exception as e:
                        print(f"Error loading {bits} model: {e}")
                        print("Trying alternative loading method...")
                        try:
                            # Alternative: Load from H5 file with weights only approach
                            h5_path = quant_dir / f'best_model_{bits}_0int.h5'
                            if h5_path.exists():
                                print(f"Loading weights from H5 file: {h5_path}")
                                # Create a dummy model structure to hold weights
                                dummy_model = self._create_dummy_model_for_weights(h5_path)
                                if dummy_model:
                                    self.models[bits] = dummy_model
                                    
                                    # Load parameters
                                    params_path = quant_dir / 'model_parameters.json'
                                    if params_path.exists():
                                        with open(params_path, 'r') as f:
                                            self.model_parameters[bits] = json.load(f)
                                else:
                                    print(f"Could not create dummy model for {bits}")
                        except Exception as e2:
                            print(f"Could not load {bits} model: {e2}")
        
        print(f"Successfully loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def _create_dummy_model_for_weights(self, h5_path):
        """
        Create a dummy model structure to hold weights loaded from H5 file
        """
        try:
            import h5py
            
            class DummyLayer:
                def __init__(self, name, weights_data):
                    self.name = name
                    self._weights = weights_data
                    self.__class__.__name__ = "DummyLayer"
                
                def get_weights(self):
                    return self._weights
            
            class DummyModel:
                def __init__(self):
                    self.layers = []
                    self._total_params = 0
                
                def count_params(self):
                    return self._total_params
            
            # Load weights from H5 file
            dummy_model = DummyModel()
            
            with h5py.File(h5_path, 'r') as f:
                # Navigate through HDF5 structure to find weights
                def extract_weights_recursive(group, prefix=""):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            # Recurse into subgroups
                            extract_weights_recursive(item, f"{prefix}{key}_")
                        elif isinstance(item, h5py.Dataset):
                            # This is a weight array
                            weight_data = item[:]
                            layer_name = f"{prefix}{key}"
                            
                            # Group weights and biases by layer
                            if 'kernel' in key.lower() or 'weight' in key.lower():
                                # Find corresponding bias
                                bias_key = key.replace('kernel', 'bias').replace('weight', 'bias')
                                bias_data = None
                                if bias_key in group:
                                    bias_data = group[bias_key][:]
                                
                                # Create dummy layer
                                layer_weights = [weight_data]
                                if bias_data is not None:
                                    layer_weights.append(bias_data)
                                
                                dummy_layer = DummyLayer(layer_name, layer_weights)
                                dummy_model.layers.append(dummy_layer)
                                dummy_model._total_params += weight_data.size
                                if bias_data is not None:
                                    dummy_model._total_params += bias_data.size
                
                # Start extraction from root or model_weights group
                if 'model_weights' in f:
                    extract_weights_recursive(f['model_weights'])
                else:
                    extract_weights_recursive(f)
            
            if len(dummy_model.layers) > 0:
                print(f"Successfully created dummy model with {len(dummy_model.layers)} layers")
                return dummy_model
            else:
                print("No weights found in H5 file")
                return None
                
        except ImportError:
            print("h5py not available - cannot load H5 files")
            return None
        except Exception as e:
            print(f"Error creating dummy model from H5: {e}")
            return None
    
    def _get_qkeras_custom_objects(self):
        """
        Get custom objects needed for loading QKeras models
        """
        try:
            from qkeras import QDense, QActivation, QConv2D
            from qkeras.quantizers import quantized_bits, quantized_relu
            
            return {
                'QDense': QDense,
                'QActivation': QActivation, 
                'QConv2D': QConv2D,
                'quantized_bits': quantized_bits,
                'quantized_relu': quantized_relu
            }
        except ImportError:
            print("QKeras not available - some models may not load properly")
            return {}
    
    def _calculate_skewness(self, data):
        """Calculate skewness manually if scipy is not available"""
        if SCIPY_AVAILABLE:
            return stats.skew(data)
        else:
            # Simple skewness calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis manually if scipy is not available"""
        if SCIPY_AVAILABLE:
            return stats.kurtosis(data)
        else:
            # Simple kurtosis calculation (excess kurtosis)
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def extract_weights_and_biases(self):
        """
        Extract weights and biases from all loaded models
        
        Returns:
            dict: Dictionary containing weights and biases for each model
        """
        weights_data = {}
        
        for model_name, model in self.models.items():
            print(f"Extracting weights from {model_name} model...")
            weights_data[model_name] = {
                'layers': {},
                'summary': {
                    'total_params': model.count_params(),
                    'trainable_params': sum([np.prod(layer.get_weights()[0].shape) 
                                           for layer in model.layers 
                                           if len(layer.get_weights()) > 0])
                }
            }
            
            for i, layer in enumerate(model.layers):
                layer_weights = layer.get_weights()
                if len(layer_weights) > 0:
                    layer_name = f"{layer.__class__.__name__}_{i}_{layer.name}"
                    
                    weights_data[model_name]['layers'][layer_name] = {}
                    
                    # Extract weights (typically index 0)
                    if len(layer_weights) >= 1:
                        weights = layer_weights[0]
                        weights_data[model_name]['layers'][layer_name]['weights'] = {
                            'values': weights.flatten(),
                            'shape': weights.shape,
                            'stats': {
                                'mean': float(np.mean(weights)),
                                'std': float(np.std(weights)),
                                'min': float(np.min(weights)),
                                'max': float(np.max(weights)),
                                'unique_values': len(np.unique(weights)),
                                'skewness': float(self._calculate_skewness(weights.flatten())),
                                'kurtosis': float(self._calculate_kurtosis(weights.flatten()))
                            }
                        }
                    
                    # Extract biases (typically index 1)
                    if len(layer_weights) >= 2:
                        biases = layer_weights[1]
                        weights_data[model_name]['layers'][layer_name]['biases'] = {
                            'values': biases.flatten(),
                            'shape': biases.shape,
                            'stats': {
                                'mean': float(np.mean(biases)),
                                'std': float(np.std(biases)),
                                'min': float(np.min(biases)),
                                'max': float(np.max(biases)),
                                'unique_values': len(np.unique(biases)),
                                'skewness': float(self._calculate_skewness(biases.flatten())),
                                'kurtosis': float(self._calculate_kurtosis(biases.flatten()))
                            }
                        }
        
        return weights_data
    
    def create_distribution_plots(self, weights_data):
        """
        Create comprehensive distribution plots
        
        Args:
            weights_data (dict): Extracted weights and biases data
        """
        print("Creating distribution plots...")
        
        # Create output directory for plots
        output_dir = self.results_dir / 'weight_distribution_analysis'
        output_dir.mkdir(exist_ok=True)
        
        # 1. Overall weight distribution comparison
        self._plot_overall_distributions(weights_data, output_dir)
        
        # 2. Layer-wise distribution comparison
        self._plot_layer_wise_distributions(weights_data, output_dir)
        
        # 3. Quantization effects analysis
        self._plot_quantization_effects(weights_data, output_dir)
        
        # 4. Statistical comparison
        self._plot_statistical_comparison(weights_data, output_dir)
        
        # 5. Unique values analysis (quantization levels)
        self._plot_unique_values_analysis(weights_data, output_dir)
        
        print(f"All plots saved to {output_dir}")
    
    def _plot_overall_distributions(self, weights_data, output_dir):
        """Plot overall weight and bias distributions for all models"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Weight and Bias Distributions Across All Models', fontsize=16)
        
        # Collect all weights and biases for each model
        all_weights = {}
        all_biases = {}
        
        for model_name, data in weights_data.items():
            weights_list = []
            biases_list = []
            
            for layer_name, layer_data in data['layers'].items():
                if 'weights' in layer_data:
                    weights_list.extend(layer_data['weights']['values'])
                if 'biases' in layer_data:
                    biases_list.extend(layer_data['biases']['values'])
            
            if weights_list:
                all_weights[model_name] = np.array(weights_list)
            if biases_list:
                all_biases[model_name] = np.array(biases_list)
        
        # Plot weight distributions
        axes[0, 0].set_title('Weight Distributions (Histogram)')
        for model_name, weights in all_weights.items():
            axes[0, 0].hist(weights, bins=50, alpha=0.6, label=model_name, density=True)
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot bias distributions
        axes[0, 1].set_title('Bias Distributions (Histogram)')
        for model_name, biases in all_biases.items():
            axes[0, 1].hist(biases, bins=50, alpha=0.6, label=model_name, density=True)
        axes[0, 1].set_xlabel('Bias Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plots for weights
        axes[1, 0].set_title('Weight Distribution Box Plots')
        weight_data_for_box = [all_weights[name] for name in all_weights.keys()]
        weight_labels = list(all_weights.keys())
        box1 = axes[1, 0].boxplot(weight_data_for_box, tick_labels=weight_labels, patch_artist=True)
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Weight Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plots for biases
        axes[1, 1].set_title('Bias Distribution Box Plots')
        bias_data_for_box = [all_biases[name] for name in all_biases.keys()]
        bias_labels = list(all_biases.keys())
        box2 = axes[1, 1].boxplot(bias_data_for_box, tick_labels=bias_labels, patch_artist=True)
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Bias Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Color the box plots
        colors = plt.cm.Set3(np.linspace(0, 1, len(weight_labels)))
        for patch, color in zip(box1['boxes'], colors):
            patch.set_facecolor(color)
        for patch, color in zip(box2['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_wise_distributions(self, weights_data, output_dir):
        """Plot layer-wise distribution comparisons"""
        
        # Get common layers across all models - match by position/index instead of exact name
        if self.non_quantized_key not in weights_data:
            print("No non-quantized model found for layer-wise comparison")
            return
        
        # Create layer mapping based on position and type
        layer_mappings = self._create_layer_mappings(weights_data)
        
        print(f"Found {len(layer_mappings)} layer groups for comparison")
        
        if not layer_mappings:
            print("No common layer groups found across models")
            return
        
        # Create plots for each layer group
        for group_name, layer_mapping in layer_mappings.items():
            self._plot_layer_group_comparison(weights_data, group_name, layer_mapping, output_dir)
    
    def _create_layer_mappings(self, weights_data):
        """Create mappings between layers across different models based on position and weights shape"""
        
        # Get the reference model (non-quantized)
        ref_model_data = weights_data[self.non_quantized_key]
        ref_layers = list(ref_model_data['layers'].keys())
        
        layer_mappings = {}
        
        # For each layer in the reference model, try to find corresponding layers in other models
        for i, ref_layer_name in enumerate(ref_layers):
            ref_layer_data = ref_model_data['layers'][ref_layer_name]
            
            if 'weights' not in ref_layer_data:
                continue
                
            ref_weights_shape = ref_layer_data['weights']['shape']
            
            # Create a group for this layer
            group_name = f"Layer_{i}_{ref_layer_name.split('_')[0]}"
            layer_mappings[group_name] = {self.non_quantized_key: ref_layer_name}
            
            # Find corresponding layers in other models by matching weight shapes
            for model_name, model_data in weights_data.items():
                if model_name == self.non_quantized_key:
                    continue
                    
                # Try to match by position first
                model_layers = list(model_data['layers'].keys())
                
                matched = False
                
                # Try position-based matching
                if i < len(model_layers):
                    candidate_layer = model_layers[i]
                    candidate_data = model_data['layers'][candidate_layer]
                    
                    if 'weights' in candidate_data:
                        candidate_shape = candidate_data['weights']['shape']
                        if candidate_shape == ref_weights_shape:
                            layer_mappings[group_name][model_name] = candidate_layer
                            matched = True
                
                # If position-based matching failed, try shape-based matching
                if not matched:
                    for candidate_layer in model_layers:
                        candidate_data = model_data['layers'][candidate_layer]
                        if 'weights' in candidate_data:
                            candidate_shape = candidate_data['weights']['shape']
                            if candidate_shape == ref_weights_shape:
                                layer_mappings[group_name][model_name] = candidate_layer
                                matched = True
                                break
        
        # Filter out groups that don't have matches in most models
        min_models = max(1, len(weights_data) // 2)  # At least half the models
        filtered_mappings = {}
        for group_name, mapping in layer_mappings.items():
            if len(mapping) >= min_models:
                filtered_mappings[group_name] = mapping
        
        return filtered_mappings
    
    def _plot_layer_group_comparison(self, weights_data, group_name, layer_mapping, output_dir):
        """Plot comparison for a specific layer group"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{group_name} - Weight and Bias Distributions', fontsize=16)
        
        # Weight distributions
        axes[0].set_title('Weights')
        for model_name, layer_name in layer_mapping.items():
            if layer_name in weights_data[model_name]['layers'] and 'weights' in weights_data[model_name]['layers'][layer_name]:
                weights = weights_data[model_name]['layers'][layer_name]['weights']['values']
                axes[0].hist(weights, bins=30, alpha=0.6, label=model_name, density=True)
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bias distributions
        axes[1].set_title('Biases')
        for model_name, layer_name in layer_mapping.items():
            if layer_name in weights_data[model_name]['layers'] and 'biases' in weights_data[model_name]['layers'][layer_name]:
                biases = weights_data[model_name]['layers'][layer_name]['biases']['values']
                axes[1].hist(biases, bins=30, alpha=0.6, label=model_name, density=True)
        axes[1].set_xlabel('Bias Value')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{group_name}_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_type_comparison(self, weights_data, layer_type, layer_names, output_dir):
        """Plot comparison for a specific layer type"""
        
        n_layers = len(layer_names)
        n_models = len(weights_data)
        
        fig, axes = plt.subplots(n_layers, 2, figsize=(16, 4 * n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{layer_type} Layer Weight and Bias Distributions', fontsize=16)
        
        for i, layer_name in enumerate(layer_names):
            # Weight distributions
            axes[i, 0].set_title(f'{layer_name} - Weights')
            for model_name, data in weights_data.items():
                if layer_name in data['layers'] and 'weights' in data['layers'][layer_name]:
                    weights = data['layers'][layer_name]['weights']['values']
                    axes[i, 0].hist(weights, bins=30, alpha=0.6, label=model_name, density=True)
            axes[i, 0].set_xlabel('Weight Value')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Bias distributions
            axes[i, 1].set_title(f'{layer_name} - Biases')
            for model_name, data in weights_data.items():
                if layer_name in data['layers'] and 'biases' in data['layers'][layer_name]:
                    biases = data['layers'][layer_name]['biases']['values']
                    axes[i, 1].hist(biases, bins=30, alpha=0.6, label=model_name, density=True)
            axes[i, 1].set_xlabel('Bias Value')
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{layer_type}_layer_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quantization_effects(self, weights_data, output_dir):
        """Plot the effects of quantization on weight distributions"""
        
        if self.non_quantized_key not in weights_data:
            print("No non-quantized model found for quantization effects analysis")
            return
        
        # Create comparison plots showing before/after quantization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quantization Effects on Weight Distributions', fontsize=16)
        
        reference_model = weights_data[self.non_quantized_key]
        
        # Get all weights and biases from reference model
        ref_weights = []
        ref_biases = []
        for layer_data in reference_model['layers'].values():
            if 'weights' in layer_data:
                ref_weights.extend(layer_data['weights']['values'])
            if 'biases' in layer_data:
                ref_biases.extend(layer_data['biases']['values'])
        
        ref_weights = np.array(ref_weights)
        ref_biases = np.array(ref_biases)
        
        # Plot 1: Weight range comparison
        axes[0, 0].set_title('Weight Value Ranges')
        model_names = []
        weight_ranges = []
        for model_name, data in weights_data.items():
            all_weights = []
            for layer_data in data['layers'].values():
                if 'weights' in layer_data:
                    all_weights.extend(layer_data['weights']['values'])
            if all_weights:
                model_names.append(model_name)
                weight_ranges.append([np.min(all_weights), np.max(all_weights)])
        
        weight_ranges = np.array(weight_ranges)
        x_pos = np.arange(len(model_names))
        axes[0, 0].bar(x_pos, weight_ranges[:, 1] - weight_ranges[:, 0], 
                       bottom=weight_ranges[:, 0], alpha=0.7)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].set_ylabel('Weight Value Range')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Unique values count (quantization levels)
        axes[0, 1].set_title('Number of Unique Weight Values')
        unique_counts = []
        for model_name in model_names:
            data = weights_data[model_name]
            all_weights = []
            for layer_data in data['layers'].values():
                if 'weights' in layer_data:
                    all_weights.extend(layer_data['weights']['values'])
            unique_counts.append(len(np.unique(all_weights)))
        
        bars = axes[0, 1].bar(model_names, unique_counts, alpha=0.7)
        axes[0, 1].set_ylabel('Number of Unique Values')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, unique_counts):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{count}', ha='center', va='bottom')
        
        # Plot 3: Weight distribution comparison (selected quantization levels)
        axes[0, 2].set_title('Weight Distributions (Selected Models)')
        selected_models = [self.non_quantized_key, '8bit', '4bit', '2bit']
        for model_name in selected_models:
            if model_name in weights_data:
                all_weights = []
                for layer_data in weights_data[model_name]['layers'].values():
                    if 'weights' in layer_data:
                        all_weights.extend(layer_data['weights']['values'])
                if all_weights:
                    axes[0, 2].hist(all_weights, bins=50, alpha=0.6, 
                                   label=model_name, density=True)
        axes[0, 2].set_xlabel('Weight Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Standard deviation comparison
        axes[1, 0].set_title('Weight Standard Deviation')
        std_values = []
        for model_name in model_names:
            data = weights_data[model_name]
            all_weights = []
            for layer_data in data['layers'].values():
                if 'weights' in layer_data:
                    all_weights.extend(layer_data['weights']['values'])
            std_values.append(np.std(all_weights))
        
        axes[1, 0].bar(model_names, std_values, alpha=0.7)
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Bias analysis
        axes[1, 1].set_title('Bias Value Ranges')
        bias_ranges = []
        bias_model_names = []
        for model_name, data in weights_data.items():
            all_biases = []
            for layer_data in data['layers'].values():
                if 'biases' in layer_data:
                    all_biases.extend(layer_data['biases']['values'])
            if all_biases:
                bias_model_names.append(model_name)
                bias_ranges.append([np.min(all_biases), np.max(all_biases)])
        
        if bias_ranges:
            bias_ranges = np.array(bias_ranges)
            x_pos = np.arange(len(bias_model_names))
            axes[1, 1].bar(x_pos, bias_ranges[:, 1] - bias_ranges[:, 0], 
                          bottom=bias_ranges[:, 0], alpha=0.7)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(bias_model_names, rotation=45)
            axes[1, 1].set_ylabel('Bias Value Range')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Quantization error (if reference model exists)
        axes[1, 2].set_title('Weight Distribution Overlap with Original')
        if self.non_quantized_key in weights_data:
            overlaps = []
            for model_name in model_names:
                if model_name != self.non_quantized_key:
                    # Calculate histogram overlap (simplified metric)
                    quant_weights = []
                    for layer_data in weights_data[model_name]['layers'].values():
                        if 'weights' in layer_data:
                            quant_weights.extend(layer_data['weights']['values'])
                    
                    if quant_weights:
                        # Calculate KL divergence or correlation
                        hist_ref, bins = np.histogram(ref_weights, bins=50, density=True)
                        hist_quant, _ = np.histogram(quant_weights, bins=bins, density=True)
                        
                        # Calculate correlation coefficient
                        correlation = np.corrcoef(hist_ref, hist_quant)[0, 1]
                        overlaps.append(correlation)
                    else:
                        overlaps.append(0)
                        
            quant_model_names = [name for name in model_names if name != self.non_quantized_key]
            axes[1, 2].bar(quant_model_names, overlaps, alpha=0.7)
            axes[1, 2].set_ylabel('Distribution Correlation')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].set_ylim([0, 1])
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quantization_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_comparison(self, weights_data, output_dir):
        """Create statistical comparison plots"""
        
        # Prepare data for statistical analysis
        stats_data = []
        
        for model_name, data in weights_data.items():
            for layer_name, layer_data in data['layers'].items():
                if 'weights' in layer_data:
                    weights = layer_data['weights']['values']
                    stats_data.append({
                        'model': model_name,
                        'layer': layer_name,
                        'parameter_type': 'weights',
                        'mean': np.mean(weights),
                        'std': np.std(weights),
                        'min': np.min(weights),
                        'max': np.max(weights),
                        'skewness': self._calculate_skewness(weights) if SCIPY_AVAILABLE else 0.0,
                        'kurtosis': self._calculate_kurtosis(weights) if SCIPY_AVAILABLE else 0.0,
                        'unique_values': len(np.unique(weights))
                    })
                
                if 'biases' in layer_data:
                    biases = layer_data['biases']['values']
                    stats_data.append({
                        'model': model_name,
                        'layer': layer_name,
                        'parameter_type': 'biases',
                        'mean': np.mean(biases),
                        'std': np.std(biases),
                        'min': np.min(biases),
                        'max': np.max(biases),
                        'skewness': self._calculate_skewness(biases) if SCIPY_AVAILABLE else 0.0,
                        'kurtosis': self._calculate_kurtosis(biases) if SCIPY_AVAILABLE else 0.0,
                        'unique_values': len(np.unique(biases))
                    })
        
        df = pd.DataFrame(stats_data)
        
        # Create statistical comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis of Weight and Bias Distributions', fontsize=16)
        
        # Plot 1: Mean values comparison
        weight_means = df[df['parameter_type'] == 'weights'].groupby('model')['mean'].mean()
        bias_means = df[df['parameter_type'] == 'biases'].groupby('model')['mean'].mean()
        
        x = np.arange(len(weight_means))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, weight_means.values, width, label='Weights', alpha=0.7)
        axes[0, 0].bar(x + width/2, bias_means.values, width, label='Biases', alpha=0.7)
        axes[0, 0].set_title('Average Mean Values')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(weight_means.index, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation comparison
        weight_stds = df[df['parameter_type'] == 'weights'].groupby('model')['std'].mean()
        bias_stds = df[df['parameter_type'] == 'biases'].groupby('model')['std'].mean()
        
        axes[0, 1].bar(x - width/2, weight_stds.values, width, label='Weights', alpha=0.7)
        axes[0, 1].bar(x + width/2, bias_stds.values, width, label='Biases', alpha=0.7)
        axes[0, 1].set_title('Average Standard Deviation')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(weight_stds.index, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Range comparison
        weight_ranges = df[df['parameter_type'] == 'weights'].groupby('model').apply(
            lambda x: (x['max'] - x['min']).mean()
        )
        bias_ranges = df[df['parameter_type'] == 'biases'].groupby('model').apply(
            lambda x: (x['max'] - x['min']).mean()
        )
        
        axes[0, 2].bar(x - width/2, weight_ranges.values, width, label='Weights', alpha=0.7)
        axes[0, 2].bar(x + width/2, bias_ranges.values, width, label='Biases', alpha=0.7)
        axes[0, 2].set_title('Average Value Range')
        axes[0, 2].set_xlabel('Model')
        axes[0, 2].set_ylabel('Value Range')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(weight_ranges.index, rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Skewness comparison
        weight_skew = df[df['parameter_type'] == 'weights'].groupby('model')['skewness'].mean()
        bias_skew = df[df['parameter_type'] == 'biases'].groupby('model')['skewness'].mean()
        
        axes[1, 0].bar(x - width/2, weight_skew.values, width, label='Weights', alpha=0.7)
        axes[1, 0].bar(x + width/2, bias_skew.values, width, label='Biases', alpha=0.7)
        axes[1, 0].set_title('Average Skewness')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Skewness')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(weight_skew.index, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Kurtosis comparison
        weight_kurt = df[df['parameter_type'] == 'weights'].groupby('model')['kurtosis'].mean()
        bias_kurt = df[df['parameter_type'] == 'biases'].groupby('model')['kurtosis'].mean()
        
        axes[1, 1].bar(x - width/2, weight_kurt.values, width, label='Weights', alpha=0.7)
        axes[1, 1].bar(x + width/2, bias_kurt.values, width, label='Biases', alpha=0.7)
        axes[1, 1].set_title('Average Kurtosis')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('Kurtosis')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(weight_kurt.index, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Unique values (quantization effect)
        weight_unique = df[df['parameter_type'] == 'weights'].groupby('model')['unique_values'].sum()
        
        axes[1, 2].bar(weight_unique.index, weight_unique.values, alpha=0.7)
        axes[1, 2].set_title('Total Unique Weight Values')
        axes[1, 2].set_xlabel('Model')
        axes[1, 2].set_ylabel('Number of Unique Values')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(weight_unique.values):
            axes[1, 2].text(i, v, f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistical summary
        summary_stats = df.groupby(['model', 'parameter_type']).agg({
            'mean': ['mean', 'std'],
            'std': ['mean', 'std'],
            'unique_values': 'sum',
            'skewness': 'mean',
            'kurtosis': 'mean'
        }).round(6)
        
        summary_stats.to_csv(output_dir / 'statistical_summary.csv')
        print(f"Statistical summary saved to {output_dir / 'statistical_summary.csv'}")
    
    def _plot_unique_values_analysis(self, weights_data, output_dir):
        """Analyze and plot unique values (quantization levels) in detail"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quantization Level Analysis (Unique Values)', fontsize=16)
        
        # Collect unique values data
        unique_data = {}
        for model_name, data in weights_data.items():
            unique_data[model_name] = {
                'total_weights': 0,
                'unique_weights': 0,
                'layer_unique_counts': []
            }
            
            for layer_name, layer_data in data['layers'].items():
                if 'weights' in layer_data:
                    weights = layer_data['weights']['values']
                    unique_data[model_name]['total_weights'] += len(weights)
                    unique_count = len(np.unique(weights))
                    unique_data[model_name]['unique_weights'] += unique_count
                    unique_data[model_name]['layer_unique_counts'].append(unique_count)
        
        # Plot 1: Total unique values per model
        models = list(unique_data.keys())
        total_unique = [unique_data[model]['unique_weights'] for model in models]
        
        bars = axes[0, 0].bar(models, total_unique, alpha=0.7)
        axes[0, 0].set_title('Total Unique Weight Values per Model')
        axes[0, 0].set_ylabel('Number of Unique Values')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, total_unique):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:,}', ha='center', va='bottom')
        
        # Plot 2: Unique values as percentage of total weights
        total_weights = [unique_data[model]['total_weights'] for model in models]
        percentages = [unique/total * 100 for unique, total in zip(total_unique, total_weights)]
        
        axes[0, 1].bar(models, percentages, alpha=0.7, color='orange')
        axes[0, 1].set_title('Unique Values as % of Total Weights')
        axes[0, 1].set_ylabel('Percentage (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Layer-wise unique value distribution
        axes[1, 0].set_title('Unique Values Distribution Across Layers')
        for i, model in enumerate(models):
            layer_counts = unique_data[model]['layer_unique_counts']
            if layer_counts:
                axes[1, 0].boxplot(layer_counts, positions=[i], widths=0.6, patch_artist=True,
                                  boxprops=dict(facecolor=f'C{i}', alpha=0.7))
        
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].set_ylabel('Unique Values per Layer')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Quantization compression ratio
        if self.non_quantized_key in unique_data:
            reference_unique = unique_data[self.non_quantized_key]['unique_weights']
            compression_ratios = []
            quant_models = []
            
            for model in models:
                if model != self.non_quantized_key:
                    ratio = reference_unique / unique_data[model]['unique_weights']
                    compression_ratios.append(ratio)
                    quant_models.append(model)
            
            if compression_ratios:
                axes[1, 1].bar(quant_models, compression_ratios, alpha=0.7, color='red')
                axes[1, 1].set_title('Quantization Compression Ratio\n(Original / Quantized Unique Values)')
                axes[1, 1].set_ylabel('Compression Ratio')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels
                for i, (model, ratio) in enumerate(zip(quant_models, compression_ratios)):
                    axes[1, 1].text(i, ratio, f'{ratio:.1f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'unique_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, weights_data):
        """Generate a comprehensive summary report"""
        
        output_dir = self.results_dir / 'weight_distribution_analysis'
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("WEIGHT DISTRIBUTION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n")
            f.write(f"Models Analyzed: {len(weights_data)}\n\n")
            
            # Model summary
            f.write("MODEL SUMMARY:\n")
            f.write("-" * 20 + "\n")
            for model_name, data in weights_data.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Total Parameters: {data['summary']['total_params']:,}\n")
                f.write(f"  Trainable Parameters: {data['summary']['trainable_params']:,}\n")
                f.write(f"  Number of Layers with Weights: {len(data['layers'])}\n")
                
                if model_name in self.model_parameters:
                    params = self.model_parameters[model_name]
                    if 'weight_bits' in params:
                        f.write(f"  Weight Bits: {params['weight_bits']}\n")
                        f.write(f"  Activation Bits: {params['activation_bits']}\n")
                        f.write(f"  Integer Bits: {params['integer_bits']}\n")
            
            # Quantization effects summary
            f.write("\n\nQUANTIZATION EFFECTS:\n")
            f.write("-" * 25 + "\n")
            
            if self.non_quantized_key in weights_data:
                ref_data = weights_data[self.non_quantized_key]
                ref_unique = sum([layer['weights']['stats']['unique_values'] 
                                for layer in ref_data['layers'].values() 
                                if 'weights' in layer])
                
                f.write(f"Original Model Unique Values: {ref_unique:,}\n\n")
                
                for model_name, data in weights_data.items():
                    if model_name != self.non_quantized_key:
                        quant_unique = sum([layer['weights']['stats']['unique_values'] 
                                          for layer in data['layers'].values() 
                                          if 'weights' in layer])
                        compression_ratio = ref_unique / quant_unique if quant_unique > 0 else 0
                        
                        f.write(f"{model_name}:\n")
                        f.write(f"  Unique Values: {quant_unique:,}\n")
                        f.write(f"  Compression Ratio: {compression_ratio:.2f}x\n")
                        f.write(f"  Size Reduction: {(1 - quant_unique/ref_unique)*100:.1f}%\n\n")
            
            # Layer analysis
            f.write("LAYER ANALYSIS:\n")
            f.write("-" * 15 + "\n")
            
            # Find common layers
            if self.non_quantized_key in weights_data:
                common_layers = set(weights_data[self.non_quantized_key]['layers'].keys())
                for data in weights_data.values():
                    common_layers &= set(data['layers'].keys())
                
                f.write(f"Common layers across all models: {len(common_layers)}\n")
                
                for layer_name in sorted(common_layers):
                    f.write(f"\n{layer_name}:\n")
                    for model_name, data in weights_data.items():
                        if 'weights' in data['layers'][layer_name]:
                            stats = data['layers'][layer_name]['weights']['stats']
                            f.write(f"  {model_name}: mean={stats['mean']:.6f}, "
                                   f"std={stats['std']:.6f}, unique={stats['unique_values']}\n")
        
        print(f"Summary report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete weight distribution analysis"""
        
        print("Starting Weight Distribution Analysis...")
        print("="*50)
        
        # Load models
        if not self.load_models():
            print("Error: Could not load any models!")
            return False
        
        # Extract weights and biases
        print("\nExtracting weights and biases...")
        weights_data = self.extract_weights_and_biases()
        
        if not weights_data:
            print("Error: No weight data extracted!")
            return False
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self.create_distribution_plots(weights_data)
        
        # Generate summary report
        print("\nGenerating summary report...")
        self.generate_summary_report(weights_data)
        
        print("\n" + "="*50)
        print("Analysis complete!")
        print(f"Results saved to: {self.results_dir / 'weight_distribution_analysis'}")
        
        return True


def main():
    """Main function to run the weight distribution analysis"""
    
    # Set the results directory path
    results_dir = "combined_results_20250730_021907"
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found!")
        print("Please make sure you're running this script from the correct directory.")
        return
    
    # Create analyzer and run analysis
    analyzer = WeightDistributionAnalyzer(results_dir)
    
    try:
        success = analyzer.run_analysis()
        if success:
            print("\n✅ Weight distribution analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()