#!/usr/bin/env python3
"""
Test script to evaluate model accuracy with quantized weights.
This compares the performance of the original model vs quantized model.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

# Add parent directory to path for imports
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)

import OptimizedDataGenerator4 as ODG
from quantized_cnn_model_final import (
    create_simple_final_model,
    quantize_model_weights,
    analyze_model_weights
)

def setup_data_generators():
    """Setup data generators."""
    BASE_DIR = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
    TRAIN_DIR = BASE_DIR / "tfrecords_train"
    VALIDATION_DIR = BASE_DIR / "tfrecords_validation"

    train_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(TRAIN_DIR),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )

    val_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(VALIDATION_DIR),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )
    
    return train_gen, val_gen

def create_quantized_model_from_weights(model, quantized_weights):
    """
    Create a new model with quantized weights applied.
    """
    # Create a copy of the model
    quantized_model = tf.keras.models.clone_model(model)
    quantized_model.set_weights(model.get_weights())  # Copy original weights first
    
    # Apply quantized weights
    for layer in quantized_model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            kernel_name = layer.name + '/kernel'
            if kernel_name in quantized_weights:
                # Set quantized kernel weights
                layer.kernel.assign(quantized_weights[kernel_name])
                
                # Set bias if available
                bias_name = layer.name + '/bias'
                if bias_name in quantized_weights:
                    layer.bias.assign(quantized_weights[bias_name])
    
    return quantized_model

def evaluate_model_performance(model, train_gen, val_gen, model_name):
    """
    Evaluate model performance on train and validation sets.
    """
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate on training set
    train_loss, train_acc = model.evaluate(train_gen, verbose=0)
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    
    return {
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'val_accuracy': val_acc,
        'val_loss': val_loss
    }

def compare_quantization_levels(model, train_gen, val_gen):
    """
    Compare different quantization levels (8-bit, 16-bit, 32-bit).
    """
    print("Comparing Different Quantization Levels")
    print("=" * 60)
    
    results = {}
    
    # Test original model (32-bit)
    print("\n1. Original Model (32-bit float)")
    results['32-bit'] = evaluate_model_performance(model, train_gen, val_gen, "Original Model")
    
    # Test 16-bit quantization
    print("\n2. 16-bit Quantized Model")
    quantized_weights_16 = quantize_model_weights(model, bits=16)
    quantized_model_16 = create_quantized_model_from_weights(model, quantized_weights_16)
    results['16-bit'] = evaluate_model_performance(quantized_model_16, train_gen, val_gen, "16-bit Quantized")
    
    # Test 8-bit quantization
    print("\n3. 8-bit Quantized Model")
    quantized_weights_8 = quantize_model_weights(model, bits=8)
    quantized_model_8 = create_quantized_model_from_weights(model, quantized_weights_8)
    results['8-bit'] = evaluate_model_performance(quantized_model_8, train_gen, val_gen, "8-bit Quantized")
    
    # Test 4-bit quantization
    print("\n4. 4-bit Quantized Model")
    quantized_weights_4 = quantize_model_weights(model, bits=4)
    quantized_model_4 = create_quantized_model_from_weights(model, quantized_weights_4)
    results['4-bit'] = evaluate_model_performance(quantized_model_4, train_gen, val_gen, "4-bit Quantized")
    
    return results

def print_comparison_table(results):
    """
    Print a comparison table of results.
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"{'Quantization':<15} {'Train Acc':<12} {'Train Loss':<12} {'Val Acc':<12} {'Val Loss':<12}")
    print("-" * 80)
    
    for quantization, metrics in results.items():
        train_acc = metrics['train_accuracy']
        train_loss = metrics['train_loss']
        val_acc = metrics['val_accuracy']
        val_loss = metrics['val_loss']
        
        print(f"{quantization:<15} {train_acc:<12.4f} {train_loss:<12.4f} {val_acc:<12.4f} {val_loss:<12.4f}")
    
    print("\nKey Observations:")
    print("• Higher bits = better accuracy but larger model size")
    print("• 8-bit usually provides good balance of size vs accuracy")
    print("• 4-bit may cause significant accuracy drop")

def analyze_weight_distributions(model, quantized_weights):
    """
    Analyze the distribution of weights before and after quantization.
    """
    print("\nWeight Distribution Analysis")
    print("=" * 50)
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            kernel_name = layer.name + '/kernel'
            if kernel_name in quantized_weights:
                original_weights = layer.kernel.numpy()
                quantized_weights_layer = quantized_weights[kernel_name]
                
                print(f"\n{layer.name}:")
                print(f"  Original - Min: {original_weights.min():.6f}, Max: {original_weights.max():.6f}, Std: {original_weights.std():.6f}")
                print(f"  Quantized - Min: {quantized_weights_layer.min():.6f}, Max: {quantized_weights_layer.max():.6f}, Std: {quantized_weights_layer.std():.6f}")
                
                # Calculate quantization error
                mse = np.mean((original_weights - quantized_weights_layer) ** 2)
                print(f"  Quantization MSE: {mse:.8f}")

def main():
    """Main evaluation function."""
    print("Quantized Model Accuracy Evaluation")
    print("=" * 50)
    
    try:
        # Setup data generators
        train_gen, val_gen = setup_data_generators()
        
        # Create and train a model (or load existing)
        print("Creating model...")
        model = create_simple_final_model()
        
        # For demonstration, we'll use random weights
        # In practice, you'd load a trained model
        print("Using model with initialized weights for demonstration...")
        
        # Analyze original weights
        print("\nOriginal weights analysis:")
        analyze_model_weights(model)
        
        # Compare different quantization levels
        results = compare_quantization_levels(model, train_gen, val_gen)
        
        # Print comparison table
        print_comparison_table(results)
        
        # Analyze weight distributions for 8-bit quantization
        quantized_weights_8 = quantize_model_weights(model, bits=8)
        analyze_weight_distributions(model, quantized_weights_8)
        
        print("\n✅ Quantization evaluation completed!")
        print("\nTo get real accuracy with trained weights:")
        print("1. Train the model first using train_final_quantized.py")
        print("2. Load the trained model")
        print("3. Run this evaluation script")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("\nNote: This script uses initialized weights for demonstration.")
        print("For real accuracy, train the model first.")

if __name__ == "__main__":
    main() 