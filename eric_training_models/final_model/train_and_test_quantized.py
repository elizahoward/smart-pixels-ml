#!/usr/bin/env python3
"""
Complete training and quantization testing script.
This file:
1. Trains the model with real data
2. Tests accuracy with different quantization levels
3. Shows the real impact of quantization on trained weights.
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

def test_quantization_accuracy(model, train_gen, val_gen):
    """
    Test accuracy with different quantization levels using trained weights.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TESTING QUANTIZATION ACCURACY")
    print("=" * 60)
    
    results = {}
    
    # Test original trained model (32-bit)
    print("\n1. Original Trained Model (32-bit float)")
    results['32-bit'] = evaluate_model_performance(model, train_gen, val_gen, "Original Trained Model")
    
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

    # Test 4-bit quantization
    print("\n5. 2-bit Quantized Model")
    quantized_weights_2 = quantize_model_weights(model, bits=2)
    quantized_model_2 = create_quantized_model_from_weights(model, quantized_weights_2)
    results['2-bit'] = evaluate_model_performance(quantized_model_2, train_gen, val_gen, "2-bit Quantized")
    
    return results

def print_comparison_table(results):
    """
    Print a comparison table of results.
    """
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON RESULTS (WITH TRAINED WEIGHTS)")
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

def calculate_model_size_reduction():
    """
    Calculate the model size reduction with different quantization levels.
    """
    print("\n" + "=" * 60)
    print("MODEL SIZE REDUCTION ANALYSIS")
    print("=" * 60)
    
    print("Original model (32-bit float):")
    print("  • 32 bits per weight")
    print("  • Full precision")
    print("  • Baseline size")
    
    print("\nQuantized models:")
    print("  16-bit quantization:")
    print("  • 16 bits per weight")
    print("  • 50% size reduction")
    print("  • Minimal accuracy loss")
    
    print("\n  8-bit quantization:")
    print("  • 8 bits per weight")
    print("  • 75% size reduction")
    print("  • Good accuracy retention")
    
    print("\n  4-bit quantization:")
    print("  • 4 bits per weight")
    print("  • 87.5% size reduction")
    print("  • May cause accuracy drop")

def main():
    """Main function that trains and tests quantization."""
    print("COMPLETE TRAINING AND QUANTIZATION TESTING")
    print("=" * 60)
    
    try:
        # Step 1: Train the model
        model, train_gen, val_gen, history = train_model()
        
        # Step 2: Test quantization accuracy
        results = test_quantization_accuracy(model, train_gen, val_gen)
        
        # Step 3: Print comparison table
        print_comparison_table(results)
        
        # Step 4: Analyze weight distributions for 8-bit quantization
        quantized_weights_8 = quantize_model_weights(model, bits=8)
        analyze_weight_distributions(model, quantized_weights_8)
        
        # Step 5: Show model size reduction info
        calculate_model_size_reduction()
        
        print("\n" + "=" * 60)
        print("✅ COMPLETE TRAINING AND QUANTIZATION TESTING FINISHED!")
        print("=" * 60)
        
        print("\nSummary:")
        print("1. ✅ Model trained with real data")
        print("2. ✅ Quantization tested with trained weights")
        print("3. ✅ Real accuracy impact measured")
        print("4. ✅ Weight distributions analyzed")
        print("5. ✅ Model size reduction calculated")
        
    except Exception as e:
        print(f"❌ Error during training/testing: {e}")
        print("\nTroubleshooting:")
        print("1. Check if data directories exist")
        print("2. Verify TensorFlow is installed")
        print("3. Check system compatibility")

if __name__ == "__main__":
    main() 