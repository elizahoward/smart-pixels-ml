#!/usr/bin/env python3
"""
Script to show how to modify the original train_and_test_quantized_with_bias.py
to use a saved model instead of training a new one every time.

This script demonstrates the key changes needed to the original script.
"""

def show_modifications():
    """
    Show the key modifications needed for the original script.
    """
    print("=" * 80)
    print("HOW TO MODIFY THE ORIGINAL TRAINING SCRIPT")
    print("=" * 80)
    
    print("\n1. ADD THESE IMPORTS TO THE ORIGINAL SCRIPT:")
    print("""
# Add these imports to train_and_test_quantized_with_bias.py
from consolidated_quantized_model import (
    create_simple_consolidated_model,
    train_consolidated_model_manual,
    analyze_model_weights,
    quantize_model_weights_and_biases,
    load_trained_model
)
""")
    
    print("\n2. REPLACE THE train_model() FUNCTION WITH:")
    print("""
def train_model():
    \"\"\"Load a saved model or train a new one if not available.\"\"\"
    print("=" * 60)
    print("STEP 1: LOADING OR TRAINING MODEL")
    print("=" * 60)
    
    # Try to load saved model first
    save_path = "smart_pixels_ml/eric_training_models/CNNquantized/saved_trained_model"
    model = load_trained_model(save_path)
    
    if model is not None:
        print("✅ Successfully loaded pre-trained model!")
        print("Model summary:")
        model.summary()
        
        # Analyze loaded model weights
        print("\\nLoaded model weights analysis:")
        analyze_model_weights(model)
        
        # Setup data generators for evaluation
        train_gen, val_gen = setup_data_generators()
        
        return model, train_gen, val_gen, None
    else:
        print("❌ No saved model found. Training a new model...")
        
        # Setup data generators
        train_gen, val_gen = setup_data_generators()
        
        # Create and train model
        model = create_simple_consolidated_model()
        
        print("Model summary:")
        model.summary()
        
        # Analyze initial weights
        print("\\nInitial weights analysis:")
        analyze_model_weights(model)
        
        # Train the model
        print("\\nTraining model...")
        history = train_consolidated_model_manual(
            model, 
            train_gen, 
            val_gen, 
            epochs=120, 
            patience=50
        )
        
        # Evaluate trained model
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        train_loss, train_acc = model.evaluate(train_gen, verbose=0)
        
        print(f"\\nTRAINED MODEL RESULTS:")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Training loss: {train_loss:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        
        # Analyze final weights
        print("\\nFinal weights analysis:")
        analyze_model_weights(model)
        
        return model, train_gen, val_gen, history
""")
    
    print("\n3. UPDATE THE main() FUNCTION CALL:")
    print("""
def main():
    \"\"\"Main function that loads/trains and tests quantization with bias quantization.\"\"\"
    print("COMPLETE TRAINING AND QUANTIZATION TESTING (WITH SAVED MODEL SUPPORT)")
    print("=" * 80)
    
    try:
        # Step 1: Load or train the model
        model, train_gen, val_gen, history = train_model()
        
        # Step 2: Test quantization accuracy with bias quantization
        results = test_quantization_accuracy_with_bias(model, train_gen, val_gen)
        
        # Step 3: Print comparison table
        print_comparison_table(results)
        
        # Step 4: Analyze weight and bias distributions for 8-bit quantization
        quantized_weights_8 = quantize_model_weights_and_biases(model, bits=8)
        analyze_weight_and_bias_distributions(model, quantized_weights_8)
        
        # Step 5: Show model size reduction info
        calculate_model_size_reduction_with_bias()
        
        print("\\n" + "=" * 80)
        print("✅ COMPLETE QUANTIZATION TESTING FINISHED!")
        print("=" * 80)
        
        print("\\nSummary:")
        print("1. ✅ Model loaded/trained successfully")
        print("2. ✅ Quantization tested with trained weights AND biases")
        print("3. ✅ Real accuracy impact measured")
        print("4. ✅ Weight and bias distributions analyzed")
        print("5. ✅ Model size reduction calculated")
        print("6. ✅ Saved model support implemented")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\\nTroubleshooting:")
        print("1. Check if data directories exist")
        print("2. Verify TensorFlow is installed")
        print("3. Check system compatibility")
        print("4. Run train_and_save_model.py first to create a saved model")
""")
    
    print("\n4. KEY BENEFITS OF THESE MODIFICATIONS:")
    print("""
✅ No more retraining every time you want to test quantization
✅ Faster iteration and experimentation
✅ Consistent model for fair comparison
✅ Easy to switch between training and loading modes
✅ Maintains all original functionality
""")
    
    print("\n5. USAGE WORKFLOW:")
    print("""
1. First time: Run train_and_save_model.py to create a saved model
2. Subsequent times: Run the modified train_and_test_quantized_with_bias.py
3. The script will automatically load the saved model
4. If no saved model exists, it will train a new one
""")

def create_modified_script():
    """
    Create a modified version of the original script.
    """
    print("\n" + "=" * 80)
    print("CREATING MODIFIED VERSION OF ORIGINAL SCRIPT")
    print("=" * 80)
    
    # Read the original script
    original_script_path = "../final_model/train_and_test_quantized_with_bias.py"
    
    try:
        with open(original_script_path, 'r') as f:
            original_content = f.read()
        
        # Create modified content
        modified_content = original_content.replace(
            "from quantized_cnn_model_final import (",
            "from quantized_cnn_model_final import (\n    # Original imports\n    create_final_quantized_cnn_model,\n    create_simple_final_model,\n    train_final_model_manual,\n    analyze_model_weights,\n    quantize_model_weights\n)\nfrom consolidated_quantized_model import (\n    # New imports for saved model support\n    create_simple_consolidated_model,\n    train_consolidated_model_manual,\n    load_trained_model\n)"
        )
        
        # Replace the train_model function
        old_train_model = """def train_model():
    \"\"\"Train the model with real data.\"\"\"
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
    print("\\nInitial weights analysis:")
    analyze_model_weights(model)
    
    # Train the model
    print("\\nTraining model...")
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
    
    print(f"\\nTRAINED MODEL RESULTS:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    
    # Analyze final weights
    print("\\nFinal weights analysis:")
    analyze_model_weights(model)
    
    return model, train_gen, val_gen, history"""
        
        new_train_model = """def train_model():
    \"\"\"Load a saved model or train a new one if not available.\"\"\"
    print("=" * 60)
    print("STEP 1: LOADING OR TRAINING MODEL")
    print("=" * 60)
    
    # Try to load saved model first
    save_path = "smart_pixels_ml/eric_training_models/CNNquantized/saved_trained_model"
    model = load_trained_model(save_path)
    
    if model is not None:
        print("✅ Successfully loaded pre-trained model!")
        print("Model summary:")
        model.summary()
        
        # Analyze loaded model weights
        print("\\nLoaded model weights analysis:")
        analyze_model_weights(model)
        
        # Setup data generators for evaluation
        train_gen, val_gen = setup_data_generators()
        
        return model, train_gen, val_gen, None
    else:
        print("❌ No saved model found. Training a new model...")
        
        # Setup data generators
        train_gen, val_gen = setup_data_generators()
        
        # Create and train model
        model = create_simple_consolidated_model()
        
        print("Model summary:")
        model.summary()
        
        # Analyze initial weights
        print("\\nInitial weights analysis:")
        analyze_model_weights(model)
        
        # Train the model
        print("\\nTraining model...")
        history = train_consolidated_model_manual(
            model, 
            train_gen, 
            val_gen, 
            epochs=120, 
            patience=50
        )
        
        # Evaluate trained model
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        train_loss, train_acc = model.evaluate(train_gen, verbose=0)
        
        print(f"\\nTRAINED MODEL RESULTS:")
        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Training loss: {train_loss:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        
        # Analyze final weights
        print("\\nFinal weights analysis:")
        analyze_model_weights(model)
        
        return model, train_gen, val_gen, history"""
        
        modified_content = modified_content.replace(old_train_model, new_train_model)
        
        # Write the modified script
        modified_script_path = "train_and_test_quantized_with_bias_modified.py"
        with open(modified_script_path, 'w') as f:
            f.write(modified_content)
        
        print(f"✅ Modified script created: {modified_script_path}")
        print("This script can now load saved models instead of training new ones!")
        
    except FileNotFoundError:
        print(f"❌ Original script not found: {original_script_path}")
        print("Make sure you're in the correct directory.")
    except Exception as e:
        print(f"❌ Error creating modified script: {e}")

if __name__ == "__main__":
    show_modifications()
    create_modified_script() 