#!/usr/bin/env python3
"""
Training script for the improved quantized CNN model.
This version should perform much better than the original quantized model.
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

# Add parent directory to path for imports
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)

import OptimizedDataGenerator4 as ODG
from quantized_cnn_model_improved import (
    create_improved_quantized_cnn_model,
    create_hybrid_quantized_model,
    create_simple_quantized_model,
    train_quantized_model_manual
)

def setup_data_generators():
    """Setup data generators similar to the notebook."""
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

def compare_models():
    """Compare different model approaches."""
    print("Comparing Different Model Approaches")
    print("=" * 50)
    
    # Setup data generators
    train_gen, val_gen = setup_data_generators()
    
    models = {
        'Improved Quantized': create_simple_quantized_model(),
        'Hybrid (Full Precision)': create_hybrid_quantized_model()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        print(f"Model parameters: {model.count_params():,}")
        model.summary()
        
        # Train the model
        history = train_quantized_model_manual(
            model, 
            train_gen, 
            val_gen, 
            epochs=30,  # Shorter for comparison
            patience=8
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        print(f"\n{model_name} results:")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        
        results[model_name] = {
            'model': model,
            'history': history,
            'val_accuracy': val_acc,
            'val_loss': val_loss
        }
    
    return results

def train_single_improved_model():
    """Train the improved quantized model."""
    print("Training improved quantized model...")
    
    # Setup data generators
    train_gen, val_gen = setup_data_generators()
    
    # Create improved model
    model = create_simple_quantized_model()
    
    print("Model summary:")
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    history = train_quantized_model_manual(
        model, 
        train_gen, 
        val_gen, 
        epochs=120, 
        patience=50
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    print(f"Final validation loss: {val_loss:.4f}")
    
    return model, history

def explain_improvements():
    """Explain the improvements made to the quantized model."""
    print("\nImprovements Made to Quantized Model")
    print("=" * 50)
    
    print("1. **Better Quantization Strategy**:")
    print("   - 8-bit Conv2D (instead of 4-bit)")
    print("   - 16-bit Dense layers (instead of 8-bit)")
    print("   - Less aggressive quantization = better gradients")
    
    print("\n2. **Improved Activation Functions**:")
    print("   - Standard ReLU instead of quantized_relu")
    print("   - Better gradient flow")
    print("   - More stable training")
    
    print("\n3. **Lighter Regularization**:")
    print("   - L2(0.001) instead of L1L2(0.01)")
    print("   - Less aggressive regularization")
    print("   - Better for quantized models")
    
    print("\n4. **Hybrid Option Available**:")
    print("   - Train with full precision")
    print("   - Quantize only for inference")
    print("   - Best of both worlds")

def main():
    """Main training function."""
    print("Improved Quantized CNN Model Training")
    print("=" * 50)
    
    try:
        # Choose training approach
        print("\nChoose training approach:")
        print("1. Train improved quantized model")
        print("2. Compare improved vs hybrid models")
        print("3. Show improvements explanation")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            model, history = train_single_improved_model()
            print("\n✅ Improved quantized model training completed!")
            
        elif choice == "2":
            results = compare_models()
            print("\n" + "="*60)
            print("COMPARISON RESULTS")
            print("="*60)
            
            for model_name, result in results.items():
                val_acc = result['val_accuracy']
                val_loss = result['val_loss']
                print(f"{model_name:<25} Acc: {val_acc:.4f} Loss: {val_loss:.4f}")
            
            print("\n✅ Model comparison completed!")
            
        elif choice == "3":
            explain_improvements()
            
        else:
            print("Invalid choice. Running improved model training...")
            model, history = train_single_improved_model()
            print("\n✅ Improved quantized model training completed!")
        
        print("\nExpected improvements:")
        print("1. Better training performance (higher accuracy)")
        print("2. Lower loss values")
        print("3. More stable training")
        print("4. Still maintains quantization benefits")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("\nTroubleshooting:")
        print("1. Check if data directories exist")
        print("2. Verify TensorFlow and QKeras are installed")
        print("3. Try the hybrid model if quantized model still has issues")

if __name__ == "__main__":
    main() 