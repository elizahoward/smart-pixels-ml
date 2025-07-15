#!/usr/bin/env python3
"""
Complete training and quantization testing script with bias quantization.
This file:
1. Trains the model with real data
2. Tests accuracy with different quantization levels (weights AND biases)
3. Shows the real impact of quantization on trained weights and biases.
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
    create_final_quantized_cnn_model,
    create_simple_final_model,
    train_final_model_manual,
    analyze_model_weights,
    quantize_model_weights
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

def quantize_weights_and_biases(weights, bits):
    """
    Enhanced quantization function that handles both weights and biases.
    """
    min_val = np.min(weights)
    max_val = np.max(weights)
    
    # Handle edge case where all weights are the same
    if max_val == min_val:
        return weights
    
    # Scale to [0, 2^bits - 1]
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((weights - min_val) * scale)
    
    # Scale back
    dequantized = quantized / scale + min_val
    
    return dequantized

def quantize_model_weights_and_biases(model, bits=8):
    """
    Quantize both model weights AND biases after training for inference.
    This is an enhanced version that quantizes biases as well.
    """
    quantized_weights = {}
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # Quantize kernel weights
            kernel_weights = layer.kernel.numpy()
            quantized_kernel = quantize_weights_and_biases(kernel_weights, bits)
            quantized_weights[layer.name + '/kernel'] = quantized_kernel
            
            # Quantize bias as well (enhanced feature)
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_weights = layer.bias.numpy()
                quantized_bias = quantize_weights_and_biases(bias_weights, bits)
                quantized_weights[layer.name + '/bias'] = quantized_bias

    return quantized_weights

def train_model():
    """Train the model with real data."""
    print("=" * 60)
    print("STEP 1: TRAINING THE MODEL")
    print("=" * 60)
    
    # Setup data generators
    train_gen, val_gen = setup_data_generators()
    
    # Create model
    model = create_simple_final_model()
    
    print("Model summary:")
    model.summary()
    
    # Analyze initial weights
    print("\nInitial weights analysis:")
    analyze_model_weights(model)
    
    # Train the model
    print("\nTraining model...")
    history = train_final_model_manual(
        model, 
        train_gen, 
        val_gen, 
        epochs=120, 
        patience=50
    )
    
    # Evaluate trained model
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    train_loss, train_acc = model.evaluate(train_gen, verbose=0)
    
    print(f"\nTRAINED MODEL RESULTS:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    
    # Analyze final weights
    print("\nFinal weights analysis:")
    analyze_model_weights(model)
    
    return model, train_gen, val_gen, history

def create_quantized_model_from_weights(model, quantized_weights):
    """
    Create a new model with quantized weights and biases applied.
    """
    # Create a copy of the model
    quantized_model = tf.keras.models.clone_model(model)
    quantized_model.set_weights(model.get_weights())  # Copy original weights first
    
    # Apply quantized weights and biases
    for layer in quantized_model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            kernel_name = layer.name + '/kernel'
            if kernel_name in quantized_weights:
                # Set quantized kernel weights
                layer.kernel.assign(quantized_weights[kernel_name])
                
                # Set quantized bias if available
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

def test_quantization_accuracy_with_bias(model, train_gen, val_gen):
    """
    Test accuracy with different quantization levels using trained weights AND biases.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TESTING QUANTIZATION ACCURACY (WEIGHTS + BIASES)")
    print("=" * 60)
    
    results = {}
    
    # Test original trained model (32-bit)
    print("\n1. Original Trained Model (32-bit float)")
    results['32-bit'] = evaluate_model_performance(model, train_gen, val_gen, "Original Trained Model")
    
    # Test 16-bit quantization
    print("\n2. 16-bit Quantized Model (Weights + Biases)")
    quantized_weights_16 = quantize_model_weights_and_biases(model, bits=16)
    quantized_model_16 = create_quantized_model_from_weights(model, quantized_weights_16)
    results['16-bit'] = evaluate_model_performance(quantized_model_16, train_gen, val_gen, "16-bit Quantized")
    
    # Test 8-bit quantization
    print("\n3. 8-bit Quantized Model (Weights + Biases)")
    quantized_weights_8 = quantize_model_weights_and_biases(model, bits=8)
    quantized_model_8 = create_quantized_model_from_weights(model, quantized_weights_8)
    results['8-bit'] = evaluate_model_performance(quantized_model_8, train_gen, val_gen, "8-bit Quantized")
    
    # Test 4-bit quantization
    print("\n4. 4-bit Quantized Model (Weights + Biases)")
    quantized_weights_4 = quantize_model_weights_and_biases(model, bits=4)
    quantized_model_4 = create_quantized_model_from_weights(model, quantized_weights_4)
    results['4-bit'] = evaluate_model_performance(quantized_model_4, train_gen, val_gen, "4-bit Quantized")

    # Test 2-bit quantization
    print("\n5. 2-bit Quantized Model (Weights + Biases)")
    quantized_weights_2 = quantize_model_weights_and_biases(model, bits=2)
    quantized_model_2 = create_quantized_model_from_weights(model, quantized_weights_2)
    results['2-bit'] = evaluate_model_performance(quantized_model_2, train_gen, val_gen, "2-bit Quantized")
    
    return results

def print_comparison_table(results):
    """
    Print a comparison table of results.
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON RESULTS (WEIGHTS + BIASES)")
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
    print("• This shows REAL impact of quantization on trained model!")
    print("• Now includes bias quantization for more complete analysis")

def analyze_weight_and_bias_distributions(model, quantized_weights):
    """
    Analyze the distribution of weights AND biases before and after quantization.
    """
    print("\nWeight and Bias Distribution Analysis")
    print("=" * 50)
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            kernel_name = layer.name + '/kernel'
            bias_name = layer.name + '/bias'
            
            if kernel_name in quantized_weights:
                original_weights = layer.kernel.numpy()
                quantized_weights_layer = quantized_weights[kernel_name]
                
                print(f"\n{layer.name} - Weights:")
                print(f"  Original - Min: {original_weights.min():.6f}, Max: {original_weights.max():.6f}, Std: {original_weights.std():.6f}")
                print(f"  Quantized - Min: {quantized_weights_layer.min():.6f}, Max: {quantized_weights_layer.max():.6f}, Std: {quantized_weights_layer.std():.6f}")
                
                # Calculate quantization error for weights
                mse_weights = np.mean((original_weights - quantized_weights_layer) ** 2)
                print(f"  Weight Quantization MSE: {mse_weights:.8f}")
                
                # Analyze bias if available
                if bias_name in quantized_weights and hasattr(layer, 'bias') and layer.bias is not None:
                    original_bias = layer.bias.numpy()
                    quantized_bias = quantized_weights[bias_name]
                    
                    print(f"  {layer.name} - Bias:")
                    print(f"    Original - Min: {original_bias.min():.6f}, Max: {original_bias.max():.6f}, Std: {original_bias.std():.6f}")
                    print(f"    Quantized - Min: {quantized_bias.min():.6f}, Max: {quantized_bias.max():.6f}, Std: {quantized_bias.std():.6f}")
                    
                    # Calculate quantization error for bias
                    mse_bias = np.mean((original_bias - quantized_bias) ** 2)
                    print(f"    Bias Quantization MSE: {mse_bias:.8f}")

def calculate_model_size_reduction_with_bias():
    """
    Calculate the model size reduction with different quantization levels (including biases).
    """
    print("\n" + "=" * 60)
    print("MODEL SIZE REDUCTION ANALYSIS (WEIGHTS + BIASES)")
    print("=" * 60)
    
    print("Original model (32-bit float):")
    print("  • 32 bits per weight and bias")
    print("  • Full precision")
    print("  • Baseline size")
    
    print("\nQuantized models (weights + biases):")
    print("  16-bit quantization:")
    print("  • 16 bits per weight and bias")
    print("  • 50% size reduction")
    print("  • Minimal accuracy loss")
    
    print("\n  8-bit quantization:")
    print("  • 8 bits per weight and bias")
    print("  • 75% size reduction")
    print("  • Good accuracy retention")
    
    print("\n  4-bit quantization:")
    print("  • 4 bits per weight and bias")
    print("  • 87.5% size reduction")
    print("  • May cause accuracy drop")
    
    print("\n  2-bit quantization:")
    print("  • 2 bits per weight and bias")
    print("  • 93.75% size reduction")
    print("  • Likely significant accuracy drop")

def compare_quantization_methods(model, train_gen, val_gen):
    """
    Compare quantization with and without bias quantization.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: WITH vs WITHOUT BIAS QUANTIZATION")
    print("=" * 60)
    
    # Test 8-bit quantization WITHOUT bias quantization (original method)
    print("\n8-bit Quantization (Weights Only):")
    quantized_weights_only = quantize_model_weights(model, bits=8)
    quantized_model_weights_only = create_quantized_model_from_weights(model, quantized_weights_only)
    results_weights_only = evaluate_model_performance(quantized_model_weights_only, train_gen, val_gen, "8-bit Weights Only")
    
    # Test 8-bit quantization WITH bias quantization (enhanced method)
    print("\n8-bit Quantization (Weights + Biases):")
    quantized_weights_and_bias = quantize_model_weights_and_biases(model, bits=8)
    quantized_model_weights_and_bias = create_quantized_model_from_weights(model, quantized_weights_and_bias)
    results_weights_and_bias = evaluate_model_performance(quantized_model_weights_and_bias, train_gen, val_gen, "8-bit Weights + Biases")
    
    print("\nComparison Results:")
    print(f"{'Method':<25} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 50)
    print(f"{'Weights Only':<25} {results_weights_only['train_accuracy']:<12.4f} {results_weights_only['val_accuracy']:<12.4f}")
    print(f"{'Weights + Biases':<25} {results_weights_and_bias['train_accuracy']:<12.4f} {results_weights_and_bias['val_accuracy']:<12.4f}")
    
    print("\nKey Insights:")
    print("• Bias quantization may have additional impact on model performance")
    print("• More aggressive quantization = smaller model but potential accuracy loss")
    print("• Trade-off between model size and accuracy")

def main():
    """Main function that trains and tests quantization with bias quantization."""
    print("COMPLETE TRAINING AND QUANTIZATION TESTING (WITH BIAS QUANTIZATION)")
    print("=" * 80)
    
    try:
        # Step 1: Train the model
        model, train_gen, val_gen, history = train_model()
        
        # Step 2: Test quantization accuracy with bias quantization
        results = test_quantization_accuracy_with_bias(model, train_gen, val_gen)
        
        # Step 3: Print comparison table
        print_comparison_table(results)
        
        # Step 4: Compare with and without bias quantization
        compare_quantization_methods(model, train_gen, val_gen)
        
        # Step 5: Analyze weight and bias distributions for 8-bit quantization
        quantized_weights_8 = quantize_model_weights_and_biases(model, bits=8)
        analyze_weight_and_bias_distributions(model, quantized_weights_8)
        
        # Step 6: Show model size reduction info
        calculate_model_size_reduction_with_bias()
        
        print("\n" + "=" * 80)
        print("✅ COMPLETE TRAINING AND QUANTIZATION TESTING FINISHED!")
        print("=" * 80)
        
        print("\nSummary:")
        print("1. ✅ Model trained with real data")
        print("2. ✅ Quantization tested with trained weights AND biases")
        print("3. ✅ Real accuracy impact measured")
        print("4. ✅ Weight and bias distributions analyzed")
        print("5. ✅ Model size reduction calculated")
        print("6. ✅ Comparison between with/without bias quantization")
        
    except Exception as e:
        print(f"❌ Error during training/testing: {e}")
        print("\nTroubleshooting:")
        print("1. Check if data directories exist")
        print("2. Verify TensorFlow is installed")
        print("3. Check system compatibility")

if __name__ == "__main__":
    main() 